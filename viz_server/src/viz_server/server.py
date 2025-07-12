"""
ROS 2 lifecycle node that listens on a ZeroMQ REP socket and republishes
incoming data as standard ROS topics so Foxglove / RViz can visualise them.
"""

from __future__ import annotations

import argparse
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List

import fasteners
import numpy as np
import rclpy
import yaml
import zmq
from rclpy.node import Node
from scipy.spatial.transform import Rotation
from rclpy.qos import QoSProfile
from sensor_msgs.msg import JointState, PointCloud2
from sensor_msgs_py import point_cloud2 as pc2
from std_msgs.msg import Header
from termcolor import cprint
from tf2_ros import StaticTransformBroadcaster
from geometry_msgs.msg import TransformStamped
from urdf_parser_py.urdf import URDF
from visualization_msgs.msg import Marker, MarkerArray

LOCK_FILE = "/tmp/viz_server.lock"
ZMQ_PORT  = 5556


class VizServer(Node):
    """
    A single-process visualisation bridge.

    Parameters
    ----------
    urdf_path : str
        Path to the robot URDF to parse (needed for joint names & ghost meshes).
    hz : float
        Internal update rate used for trajectory interpolation.
    segment_dur : float
        Default seconds spent between two successive trajectory waypoints.
    """

    def __init__(self, urdf_path: str, hz: float = 30.0, segment_dur: float = 1.0):
        super().__init__("viz_server")

        # ------------------------------------------------------------------ #
        # singleton lock so only one server owns the port
        # ------------------------------------------------------------------ #
        self.lock = fasteners.InterProcessLock(LOCK_FILE)
        if not self.lock.acquire(blocking=False):
            cprint("Another viz_server is already running – aborting", "red")
            sys.exit(1)

        # ------------------------------------------------------------------ #
        # URDF parsing -> movable joint list
        # ------------------------------------------------------------------ #
        self.urdf_path: str    = urdf_path
        self.robot: URDF       = URDF.from_xml_file(urdf_path)
        self.joint_names: list = [j.name for j in self.robot.joints if j.type != "fixed"]
        self.num_dof: int      = len(self.joint_names)
        cprint(f"Robot DoF ({self.num_dof}): {self.joint_names}", "cyan")
        
        # ------------------------------------------------------------------ #
        # Load link config for base link name
        # ------------------------------------------------------------------ #
        self.base_link_name: str = self._load_base_link_name()
        if self.base_link_name == "error":
            cprint("Error: base link name not found", "red")
            sys.exit(1)
        cprint(f"Base link: {self.base_link_name}", "cyan")
        self.eef_base_link_name: str = self._load_eef_base_link_name()
        if self.eef_base_link_name == "error":
            cprint("Error: eef base link name not found", "red")
            sys.exit(1)
        cprint(f"End effector base link: {self.eef_base_link_name}", "cyan")
        self.eef_visual_links: list = self._load_eef_visual_links()
        if self.eef_visual_links == []:
            cprint("Error: eef visual links not found", "red")
            sys.exit(1)
        cprint(f"End effector visual links: {self.eef_visual_links}", "cyan")
        
        # Parse mimic joint relationships
        self.mimic_joints = self._parse_mimic_joints()
        if self.mimic_joints:
            cprint(f"Mimic joints found: {self.mimic_joints}", "cyan")

        # ------------------------------------------------------------------ #
        # ROS publishers
        # ------------------------------------------------------------------ #
        qos = QoSProfile(depth=1)
        self.js_pub     = self.create_publisher(JointState,   "/joint_states",  qos)
        self.marker_pub = self.create_publisher(MarkerArray,  "/viz/markers",   qos)
        
        # Multiple point cloud publishers for different types
        self.pc_robot_pub    = self.create_publisher(PointCloud2, "/viz/robot_points",    qos)
        self.pc_target_pub   = self.create_publisher(PointCloud2, "/viz/target_points",   qos)
        self.pc_obstacle_pub = self.create_publisher(PointCloud2, "/viz/obstacle_points", qos)

        # Static transform broadcaster for world frame
        self.static_tf_broadcaster = StaticTransformBroadcaster(self)
        
        # tunables
        self.rate_hz          = hz
        self.segment_duration = segment_dur
        self.rsp_proc         = None         # robot_state_publisher subprocess
        self.latest_joint_state = {}         # Track latest joint values

        # ------------------------------------------------------------------ #
        # Start robot_state_publisher immediately
        # ------------------------------------------------------------------ #
        cprint("Starting robot_state_publisher...", "green")
        xml = Path(self.urdf_path).read_text()
        self.rsp_proc = subprocess.Popen(
            [
                "ros2", "run", "robot_state_publisher", "robot_state_publisher",
                "--ros-args", "-p", f"robot_description:={xml}"
            ],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        
        # Wait a bit for robot_state_publisher to start, then publish neutral state
        threading.Thread(target=self._publish_neutral_state, daemon=True).start()
        
        # Publish static transform from world to base link
        self._publish_world_to_base_transform()

        # ------------------------------------------------------------------ #
        # ZeroMQ REP socket
        # ------------------------------------------------------------------ #
        self.zmq_ctx = zmq.Context.instance()
        self.sock    = self.zmq_ctx.socket(zmq.REP)
        self.sock.bind(f"tcp://127.0.0.1:{ZMQ_PORT}")
        threading.Thread(target=self._zmq_loop, daemon=True).start()

    def __del__(self):
        """Clean up when the node is destroyed."""
        self._cleanup()

    # ====================================================================== #
    # ZeroMQ – request loop
    # ====================================================================== #
    def _zmq_loop(self) -> None:
        """Blocking REP loop; one JSON + optional binary frame per request."""
        while rclpy.ok():
            try:
                hdr     = self.sock.recv_json()
                payload = self.sock.recv() if self.sock.getsockopt(zmq.RCVMORE) else None
                match hdr.get("cmd"):
                    case "ping":       self.sock.send_json({"status": "ok"})
                    case "joints":     self._handle_joints(hdr)
                    case "trajectory": self._handle_trajectory(hdr)
                    case "pointcloud": self._handle_pointcloud(hdr, payload)
                    case "ghost_end_effector": self._handle_ghost_end_effector(hdr)
                    case "ghost_robot": self._handle_ghost_robot(hdr)
                    case "clear_ghost_end_effector": self._handle_clear_ghost_end_effector(hdr)
                    case "clear_ghost_robot": self._handle_clear_ghost_robot(hdr)
                    case "shutdown":   self._handle_shutdown(hdr)
                    case _:            self.sock.send_json({"status": "error", "msg": "unknown cmd"})
            except Exception as exc:     # noqa: BLE001
                self.get_logger().error(str(exc))
                self.sock.send_json({"status": "error", "msg": str(exc)})

    # ====================================================================== #
    # Command handlers
    # ====================================================================== #
    def _handle_joints(self, hdr: Dict) -> None:
        """Publish one JointState message."""
        joints = hdr["joints"]
        
        # Apply mimic joint resolution first
        resolved_joints = self._resolve_mimic_joints(joints)
        
        # Now check if we have all required joints after mimic resolution
        if set(resolved_joints) != set(self.joint_names):
            missing = set(self.joint_names) - set(resolved_joints)
            extra = set(resolved_joints) - set(self.joint_names)
            error_msg = f"joint set mismatch after mimic resolution. Missing: {missing}, Extra: {extra}"
            self.sock.send_json({"status": "error", "msg": error_msg})
            return
            
        self._publish_joints(joints)  # Use original joints, _publish_joints will resolve mimics again
        self.sock.send_json({"status": "ok"})
    
    def _publish_joints(self, joints: Dict[str, float]) -> None:
        """Publish joint state without ZMQ response (for internal use)."""
        # Apply mimic joint relationships before storing
        resolved_joints = self._resolve_mimic_joints(joints)
        
        # Store latest joint state for FK calculations
        self.latest_joint_state = resolved_joints.copy()
        
        js               = JointState()
        js.header.stamp  = self.get_clock().now().to_msg()
        js.name          = self.joint_names
        js.position      = [resolved_joints[n] for n in self.joint_names]
        self.js_pub.publish(js)

    def _handle_trajectory(self, hdr: Dict) -> None:
        """Launch background interpolation thread."""
        waypoints   = hdr["waypoints"]
        segment_dur = hdr.get("segment_duration", self.segment_duration)
        rate_hz     = hdr.get("rate_hz", self.rate_hz)
        threading.Thread(target=self._run_traj,
                         args=(waypoints, segment_dur, rate_hz), daemon=True).start()
        self.sock.send_json({"status": "ok"})

    def _run_traj(self, wps: List[Dict[str, float]], seg_dur: float, rate_hz: float | None = None) -> None:
        """Simple linear interpolation at specified rate."""
        if rate_hz is None:
            rate_hz = self.rate_hz
        rate = self.create_rate(rate_hz)
        for i in range(len(wps) - 1):
            a, b  = wps[i], wps[i+1]
            steps = max(1, int(rate_hz * seg_dur))
            for s in range(steps + 1):
                alpha  = s / steps
                interp = {jn: (1-alpha)*a[jn] + alpha*b[jn] for jn in self.joint_names}
                self._publish_joints(interp)
                rate.sleep()
        time.sleep(1.0)

    def _handle_pointcloud(self, hdr: Dict, payload: bytes | None) -> None:
        """Convert raw XYZ buffer => PointCloud2."""
        if payload is None:
            self.sock.send_json({"status": "error", "msg": "missing payload"}); return
        
        pts     = np.frombuffer(payload, hdr["dtype"]).reshape(hdr["shape"])   # type: ignore[arg-type]
        frame   = hdr.get("frame", self.base_link_name)  # Default to base link
        pc_type = hdr.get("pc_type", "robot_points")     # Default type
        
        # Create simple XYZ point cloud
        header = Header(frame_id=frame, stamp=self.get_clock().now().to_msg())
        cloud  = pc2.create_cloud_xyz32(header, pts)
        
        # Publish to the appropriate topic based on type
        if pc_type == "robot_points":
            self.pc_robot_pub.publish(cloud)
        elif pc_type == "target_points":
            self.pc_target_pub.publish(cloud)
        elif pc_type == "obstacle_points":
            self.pc_obstacle_pub.publish(cloud)
        else:
            self.sock.send_json({"status": "error", "msg": f"unknown pc_type: {pc_type}"}); return
            
        self.sock.send_json({"status": "ok"})

    def _handle_ghost_end_effector(self, hdr: Dict) -> None:
        """Spawn a translucent mesh for a URDF link."""
        link_names = self.eef_visual_links
        pose = hdr["pose"]
        color = hdr.get("color", [0, 1, 0]); scale = hdr.get("scale", 1.0)
        alpha = hdr.get("alpha", 0.5)

        # Ensure we have properly resolved mimic joints in latest_joint_state
        if self.latest_joint_state:
            # Make sure we have all joints with default values for missing ones
            complete_joint_state = {}
            for joint_name in self.joint_names:
                complete_joint_state[joint_name] = self.latest_joint_state.get(joint_name, 0.0)
            self.latest_joint_state = self._resolve_mimic_joints(complete_joint_state)

        markers = []

        link = self.robot.link_map.get(self.eef_base_link_name)
        if not (link and link.visual and hasattr(link.visual, 'geometry') and link.visual.geometry and hasattr(link.visual.geometry, 'filename')):
            self.sock.send_json({"status": "error", "msg": "link/mesh not found"}); return
        mesh_uri = link.visual.geometry.filename

        # Apply visual transform for the base link
        base_pose = pose[:]
        if link.visual and hasattr(link.visual, 'origin') and link.visual.origin:
            visual_transform = self._get_visual_transform(link.visual.origin)
            base_pose = self._apply_transform(base_pose, visual_transform)

        m = Marker()
        m.header.frame_id = self.base_link_name; m.header.stamp = self.get_clock().now().to_msg()
        m.ns   = f"ghost_ee_{self.eef_base_link_name}"
        m.type = Marker.MESH_RESOURCE; m.mesh_resource = mesh_uri; m.action = Marker.ADD
        m.pose.position.x, m.pose.position.y, m.pose.position.z = base_pose[:3]
        m.pose.orientation.x, m.pose.orientation.y, m.pose.orientation.z, m.pose.orientation.w = base_pose[3:]
        m.scale.x = m.scale.y = m.scale.z = scale
        m.color.r = float(color[0])
        m.color.g = float(color[1])
        m.color.b = float(color[2])
        m.color.a = float(alpha)

        markers.append(m)

        for link_name in link_names:
            if link_name == self.eef_base_link_name:
                continue

            # Find link pose based on robot description and current joint state
            link = self.robot.link_map.get(link_name)
            if not link:
                self.sock.send_json({"status": "error", "msg": f"link {link_name} not found"}); return
            
            # Compute relative transform from eef_base_link to this link
            relative_transform = self._compute_link_transform(self.eef_base_link_name, link_name)
            
            # Apply relative transform to the base pose  
            link_pose = self._apply_transform(pose, relative_transform)
            
            # Apply visual transform for this link
            if link.visual and hasattr(link.visual, 'origin') and link.visual.origin:
                visual_transform = self._get_visual_transform(link.visual.origin)
                link_pose = self._apply_transform(link_pose, visual_transform)
            
            # Get mesh for this link
            if not (link.visual and hasattr(link.visual, 'geometry') and link.visual.geometry and hasattr(link.visual.geometry, 'filename')):
                continue  # Skip links without visual mesh
            link_mesh_uri = link.visual.geometry.filename
            
            m = Marker()
            m.header.frame_id = self.base_link_name; m.header.stamp = self.get_clock().now().to_msg()
            m.ns   = f"ghost_ee_{link_name}"
            m.type = Marker.MESH_RESOURCE; m.mesh_resource = link_mesh_uri; m.action = Marker.ADD
            m.pose.position.x, m.pose.position.y, m.pose.position.z = link_pose[:3]
            m.pose.orientation.x, m.pose.orientation.y, m.pose.orientation.z, m.pose.orientation.w = link_pose[3:]
            m.scale.x = m.scale.y = m.scale.z = scale
            m.color.r = float(color[0])
            m.color.g = float(color[1])
            m.color.b = float(color[2])
            m.color.a = float(alpha)

            markers.append(m)

        self.marker_pub.publish(MarkerArray(markers=markers))
        
        self.sock.send_json({"status": "ok"})

    def _handle_ghost_robot(self, hdr: Dict) -> None:
        """Display a translucent mesh of the entire robot at a given configuration."""
        configuration = hdr["configuration"]
        color = hdr.get("color", [0, 1, 0])
        scale = hdr.get("scale", 1.0)
        alpha = hdr.get("alpha", 0.5)

        # Validate and resolve joints
        resolved_joints = self._resolve_mimic_joints(configuration)
        
        # Check if we have all required joints
        if set(resolved_joints) != set(self.joint_names):
            missing = set(self.joint_names) - set(resolved_joints)
            extra = set(resolved_joints) - set(self.joint_names)
            error_msg = f"joint set mismatch after mimic resolution. Missing: {missing}, Extra: {extra}"
            self.sock.send_json({"status": "error", "msg": error_msg})
            return

        # Store the configuration as the current joint state for FK calculations
        self.latest_joint_state = resolved_joints.copy()

        markers = []
        timestamp = self.get_clock().now().to_msg()
        
        # Iterate through all robot links with visual geometry
        for link_name, link in self.robot.link_map.items():
            # Skip links without visual geometry
            if not (link.visual and hasattr(link.visual, 'geometry') and 
                    link.visual.geometry and hasattr(link.visual.geometry, 'filename')):
                continue
                
            # Get mesh URI
            mesh_uri = link.visual.geometry.filename
            
            # Compute transform from base link to this link
            link_transform = self._compute_link_transform(self.base_link_name, link_name)
            
            # Convert transform to pose [x, y, z, qx, qy, qz, qw]
            position = link_transform[:3, 3]
            rotation = Rotation.from_matrix(link_transform[:3, :3])
            quat = rotation.as_quat()  # [qx, qy, qz, qw]
            link_pose = [position[0], position[1], position[2], quat[0], quat[1], quat[2], quat[3]]
            
            # Apply visual transform for this link
            if link.visual and hasattr(link.visual, 'origin') and link.visual.origin:
                visual_transform = self._get_visual_transform(link.visual.origin)
                link_pose = self._apply_transform(link_pose, visual_transform)
            
            # Create marker for this link
            m = Marker()
            m.header.frame_id = self.base_link_name
            m.header.stamp = timestamp
            m.ns = f"ghost_robot_{link_name}"
            m.type = Marker.MESH_RESOURCE
            m.mesh_resource = mesh_uri
            m.action = Marker.ADD
            m.pose.position.x, m.pose.position.y, m.pose.position.z = link_pose[:3]
            m.pose.orientation.x, m.pose.orientation.y, m.pose.orientation.z, m.pose.orientation.w = link_pose[3:]
            m.scale.x = m.scale.y = m.scale.z = scale
            m.color.r = float(color[0])
            m.color.g = float(color[1])
            m.color.b = float(color[2])
            m.color.a = float(alpha)
            
            markers.append(m)

        # Publish all markers
        self.marker_pub.publish(MarkerArray(markers=markers))
        
        self.sock.send_json({"status": "ok"})

    def _handle_clear_ghost_end_effector(self, hdr: Dict) -> None:
        """Clear all ghost end effector markers."""
        # Create DELETE markers for each end effector link namespace
        markers = []
        timestamp = self.get_clock().now().to_msg()
        
        # Clear base end effector link
        m = Marker()
        m.header.frame_id = self.base_link_name
        m.header.stamp = timestamp
        m.ns = f"ghost_ee_{self.eef_base_link_name}"
        m.action = Marker.DELETEALL
        markers.append(m)
        
        # Clear all end effector visual links
        for link_name in self.eef_visual_links:
            if link_name == self.eef_base_link_name:
                continue
            m = Marker()
            m.header.frame_id = self.base_link_name
            m.header.stamp = timestamp
            m.ns = f"ghost_ee_{link_name}"
            m.action = Marker.DELETEALL
            markers.append(m)
        
        self.marker_pub.publish(MarkerArray(markers=markers))
        self.sock.send_json({"status": "ok"})

    def _handle_clear_ghost_robot(self, hdr: Dict) -> None:
        """Clear all ghost robot markers."""
        # Create DELETE markers for each robot link namespace
        markers = []
        timestamp = self.get_clock().now().to_msg()
        
        # Clear all robot links
        for link_name in self.robot.link_map.keys():
            m = Marker()
            m.header.frame_id = self.base_link_name
            m.header.stamp = timestamp
            m.ns = f"ghost_robot_{link_name}"
            m.action = Marker.DELETEALL
            markers.append(m)
        
        self.marker_pub.publish(MarkerArray(markers=markers))
        self.sock.send_json({"status": "ok"})

    def _handle_shutdown(self, hdr: Dict) -> None:
        """Handle shutdown command - clean up and exit."""
        cprint("Shutdown command received", "yellow")
        self.sock.send_json({"status": "ok"})
        
        # Give ZMQ time to send the response
        time.sleep(0.1)
        
        self._cleanup()
        
        # Force exit
        import os
        os._exit(0)

    def _cleanup(self) -> None:
        """Clean up resources."""
        cprint("Cleaning up viz_server...", "yellow")
        
        # First, close ZMQ resources to stop incoming requests
        if hasattr(self, 'sock'):
            try:
                self.sock.close(0)
            except:
                pass
        if hasattr(self, 'zmq_ctx'):
            try:
                self.zmq_ctx.term()
            except:
                pass
        
        # Now handle the subprocess
        if hasattr(self, 'rsp_proc') and self.rsp_proc:
            cprint("Terminating robot_state_publisher...", "yellow")
            try:
                # Check if process is still running
                if self.rsp_proc.poll() is None:
                    # First try graceful termination
                    self.rsp_proc.terminate()
                    
                    # Wait up to 2 seconds for graceful shutdown
                    try:
                        self.rsp_proc.wait(timeout=2.0)
                        cprint("robot_state_publisher terminated gracefully", "green")
                    except subprocess.TimeoutExpired:
                        # Force kill if it doesn't respond
                        cprint("Force killing robot_state_publisher...", "red")
                        self.rsp_proc.kill()
                        try:
                            self.rsp_proc.wait(timeout=1.0)
                            cprint("robot_state_publisher force killed", "yellow")
                        except subprocess.TimeoutExpired:
                            cprint("robot_state_publisher kill timed out", "red")
                            # Try system-level kill as last resort
                            try:
                                import os
                                os.kill(self.rsp_proc.pid, 9)
                                cprint("robot_state_publisher killed with SIGKILL", "red")
                            except:
                                cprint("Failed to kill robot_state_publisher", "red")
                else:
                    cprint("robot_state_publisher already terminated", "green")
                    
            except Exception as e:
                cprint(f"Error terminating robot_state_publisher: {e}", "red")
        
        # Final cleanup of any remaining robot_state_publisher processes via pkill
        try:
            result = subprocess.run(["pkill", "-f", "robot_state_publisher"], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                cprint("Killed remaining robot_state_publisher processes", "yellow")
        except:
            pass
                
        # Release lock
        if hasattr(self, 'lock'):
            try:
                self.lock.release()
            except:
                pass

    def _publish_neutral_state(self) -> None:
        """Publish neutral joint state (all joints at 0 radians) after startup."""
        time.sleep(1.0)  # Wait for robot_state_publisher to be ready
        neutral_joints = {name: 0.0 for name in self.joint_names}
        self._publish_joints(neutral_joints)
        cprint("Published neutral joint state (all joints at 0 radians)", "green")

    def _publish_world_to_base_transform(self) -> None:
        """Publish static transform from world frame to robot base link."""
        transform = TransformStamped()
        
        # Set header
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = "world"
        transform.child_frame_id = self.base_link_name
        
        # Identity transform (no translation or rotation)
        transform.transform.translation.x = 0.0
        transform.transform.translation.y = 0.0
        transform.transform.translation.z = 0.0
        transform.transform.rotation.x = 0.0
        transform.transform.rotation.y = 0.0
        transform.transform.rotation.z = 0.0
        transform.transform.rotation.w = 1.0
        
        # Send the transform
        self.static_tf_broadcaster.sendTransform(transform)
        cprint(f"Published static transform: world -> {self.base_link_name}", "green")

    def _load_base_link_name(self) -> str:
        """Load base link name from link_config.yaml in the same directory as URDF."""
        try:
            urdf_dir = Path(self.urdf_path).parent
            config_path = urdf_dir / "link_config.yaml"
            
            if not config_path.exists():
                cprint(f"Error: {config_path} not found","red")
                return "error"
            
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            base_link = config.get("link_config", {}).get("base_link_name", "error")
            return base_link
            
        except Exception as e:
            cprint(f"Error loading link_config.yaml: {e}, using default 'base_link'", "yellow")
            return "base_link"

    def _load_eef_base_link_name(self) -> str:
        """Load end effector base link name from link_config.yaml in the same directory as URDF."""
        try:
            urdf_dir = Path(self.urdf_path).parent
            config_path = urdf_dir / "link_config.yaml"
            
            if not config_path.exists():
                cprint(f"Error: {config_path} not found","red")
                return "error"
            
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            eef_base_link_name = config.get("link_config", {}).get("eef_base_link_name", "error")
            return eef_base_link_name

        except Exception as e:
            cprint(f"Error loading link_config.yaml: {e}, returning default 'error'", "red")
            return "error"

    def _load_eef_visual_links(self) -> list:
        """Load base link name from link_config.yaml in the same directory as URDF."""
        try:
            urdf_dir = Path(self.urdf_path).parent
            config_path = urdf_dir / "link_config.yaml"
            
            if not config_path.exists():
                cprint(f"Error: {config_path} not found","red")
                return []
            
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            eef_visual_links = config.get("link_config", {}).get("eef_visual_links", [])
            return eef_visual_links
            
        except Exception as e:
            cprint(f"Error loading link_config.yaml: {e}, returning default '[]'", "red")
            return []

    def _compute_link_transform(self, from_link: str, to_link: str) -> np.ndarray:
        """
        Compute transform from from_link to to_link using current joint states.
        
        This shows how URDF traversal works:
        1. Find path between links through URDF tree
        2. Apply joint values along the path 
        3. Compose transforms
        
        Returns: 4x4 homogeneous transform matrix
        """
        if from_link == to_link:
            return np.eye(4)
        
        # Find kinematic path from from_link to to_link
        path = self._find_kinematic_path(from_link, to_link)
        if not path:
            cprint(f"No kinematic path from {from_link} to {to_link}", "red")
            return np.eye(4)
        
        # Compute transform along the path
        transform = np.eye(4)
        
        for i in range(len(path) - 1):
            parent_link = path[i]
            child_link = path[i + 1]
            
            # Find joint connecting parent to child
            joint = None
            for j in self.robot.joints:
                if j.parent == parent_link and j.child == child_link:
                    joint = j
                    break
            
            if not joint:
                continue
                
            # Get joint transform
            joint_transform = self._get_joint_transform(joint)
            transform = transform @ joint_transform
            
        return transform
    
    def _find_kinematic_path(self, from_link: str, to_link: str) -> List[str]:
        """
        Find kinematic path between two links in URDF tree.
        
        This is how you traverse the URDF:
        - Use joint.parent and joint.child to build the tree
        - Find path using graph traversal
        """
        # Build adjacency graph from joints
        graph = {}
        for joint in self.robot.joints:
            if joint.parent not in graph:
                graph[joint.parent] = []
            if joint.child not in graph:
                graph[joint.child] = []
            graph[joint.parent].append(joint.child)
            graph[joint.child].append(joint.parent)  # Bidirectional
        
        # BFS to find path
        if from_link not in graph or to_link not in graph:
            return []
            
        queue = [(from_link, [from_link])]
        visited = set()
        
        while queue:
            current, path = queue.pop(0)
            if current == to_link:
                return path
                
            if current in visited:
                continue
            visited.add(current)
            
            for neighbor in graph.get(current, []):
                if neighbor not in visited:
                    queue.append((neighbor, path + [neighbor]))
        
        return []  # No path found
    
    def _get_joint_transform(self, joint) -> np.ndarray:
        """
        Get 4x4 transform for a joint given current joint state.
        
        This shows how to apply joint values:
        1. Start with joint.origin (fixed transform)
        2. Apply current joint value based on joint type and axis
        """
        # Start with joint's fixed transform (origin)
        transform = np.eye(4)
        
        if hasattr(joint, 'origin') and joint.origin:
            # Apply translation
            if hasattr(joint.origin, 'xyz') and joint.origin.xyz:
                transform[:3, 3] = joint.origin.xyz
            
            # Apply rotation (RPY to rotation matrix)
            if hasattr(joint.origin, 'rpy') and joint.origin.rpy:
                r = Rotation.from_euler('xyz', joint.origin.rpy)
                transform[:3, :3] = r.as_matrix()
        
        # Apply current joint value
        if joint.name in self.latest_joint_state:
            joint_value = self.latest_joint_state[joint.name]
            
            if joint.type == 'revolute' or joint.type == 'continuous':
                # Rotation around axis
                if hasattr(joint, 'axis') and joint.axis:
                    axis = np.array(joint.axis)
                    axis_rotation = Rotation.from_rotvec(joint_value * axis)
                    # Apply rotation to existing transform
                    joint_rot = np.eye(4)
                    joint_rot[:3, :3] = axis_rotation.as_matrix()
                    transform = transform @ joint_rot
                    
            elif joint.type == 'prismatic':
                # Translation along axis  
                if hasattr(joint, 'axis') and joint.axis:
                    axis = np.array(joint.axis)
                    translation = joint_value * axis
                    joint_trans = np.eye(4)
                    joint_trans[:3, 3] = translation
                    transform = transform @ joint_trans
        
        return transform
    
    def _apply_transform(self, base_pose: List[float], transform: np.ndarray) -> List[float]:
        """
        Apply a 4x4 transform to a pose [x,y,z,qx,qy,qz,qw].
        
        Returns transformed pose in same format.
        """
        # Convert pose to 4x4 matrix
        base_matrix = np.eye(4)
        base_matrix[:3, 3] = base_pose[:3]  # translation
        quat = base_pose[3:]  # [qx, qy, qz, qw]
        base_matrix[:3, :3] = Rotation.from_quat(quat).as_matrix()
        
        # Apply transform
        result_matrix = base_matrix @ transform
        
        # Convert back to pose format
        position = result_matrix[:3, 3]
        rotation = Rotation.from_matrix(result_matrix[:3, :3])
        quat = rotation.as_quat()  # [qx, qy, qz, qw]
        
        return [position[0], position[1], position[2], quat[0], quat[1], quat[2], quat[3]]

    def _parse_mimic_joints(self) -> Dict[str, Dict]:
        """
        Parse mimic joint relationships from URDF.
        
        Returns dict: {mimic_joint_name: {parent: str, multiplier: float, offset: float}}
        """
        mimic_joints = {}
        
        for joint in self.robot.joints:
            if hasattr(joint, 'mimic') and joint.mimic:
                # Handle the case where multiplier might be None
                multiplier = getattr(joint.mimic, 'multiplier', None)
                if multiplier is None:
                    multiplier = 1.0
                
                offset = getattr(joint.mimic, 'offset', None)
                if offset is None:
                    offset = 0.0
                
                mimic_info = {
                    'parent': joint.mimic.joint,
                    'multiplier': multiplier,
                    'offset': offset
                }
                mimic_joints[joint.name] = mimic_info
                
        return mimic_joints

    def _resolve_mimic_joints(self, joints: Dict[str, float]) -> Dict[str, float]:
        """
        Apply mimic joint relationships to resolve all joint values.
        
        If both parent and mimic joint are specified, parent takes precedence.
        If only mimic joint is specified, we compute parent value (reverse mimic).
        """
        resolved = joints.copy()
        
        for mimic_joint, mimic_info in self.mimic_joints.items():
            parent_joint = mimic_info['parent']
            multiplier = mimic_info['multiplier']
            offset = mimic_info['offset']
            
            parent_in_joints = parent_joint in resolved
            mimic_in_joints = mimic_joint in resolved
            
            if parent_in_joints and mimic_in_joints:
                # Both specified - check for consistency and use parent
                parent_val = resolved[parent_joint]
                mimic_val = resolved[mimic_joint]
                expected_mimic = parent_val * multiplier + offset
                
                if abs(mimic_val - expected_mimic) > 1e-6:
                    cprint(f"Warning: Inconsistent mimic joint values for {mimic_joint}. " +
                          f"Expected {expected_mimic:.4f} from {parent_joint}={parent_val:.4f}, " +
                          f"got {mimic_val:.4f}. Using parent value.", "yellow")
                
                resolved[mimic_joint] = expected_mimic
                
            elif parent_in_joints and not mimic_in_joints:
                # Only parent specified - compute mimic
                resolved[mimic_joint] = resolved[parent_joint] * multiplier + offset
                
            elif not parent_in_joints and mimic_in_joints:
                # Only mimic specified - compute parent (reverse mimic)
                if multiplier != 0:
                    resolved[parent_joint] = (resolved[mimic_joint] - offset) / multiplier
                else:
                    cprint(f"Warning: Cannot reverse mimic for {mimic_joint} (multiplier=0)", "yellow")
                    
            elif not parent_in_joints and not mimic_in_joints:
                # Neither specified - this will be caught by the joint set validation
                pass
                
        return resolved

    def _get_visual_transform(self, visual_origin) -> np.ndarray:
        """
        Get 4x4 transform from a visual origin element.
        
        Visual origins specify how the mesh should be transformed relative to the link frame.
        """
        transform = np.eye(4)
        
        # Apply translation
        if hasattr(visual_origin, 'xyz') and visual_origin.xyz:
            transform[:3, 3] = visual_origin.xyz
        
        # Apply rotation (RPY to rotation matrix)
        if hasattr(visual_origin, 'rpy') and visual_origin.rpy:
            r = Rotation.from_euler('xyz', visual_origin.rpy)
            transform[:3, :3] = r.as_matrix()
        
        return transform

# ---------------------------------------------------------------------- #
# CLI entry-point
# ---------------------------------------------------------------------- #
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--urdf", required=True, help="Path to robot URDF")
    p.add_argument("--hz",   type=float, default=30.0, help="Interpolator rate [Hz]")
    p.add_argument("--segment_duration", type=float, default=1.0,
                   help="Seconds per trajectory segment")
    args = p.parse_args()

    rclpy.init()
    
    # Create the server
    server = VizServer(args.urdf, args.hz, args.segment_duration)
    
    # Set up signal handling for graceful shutdown
    def signal_handler(signum, frame):
        cprint(f"Received signal {signum}, shutting down gracefully...", "yellow")
        server._cleanup()
        rclpy.shutdown()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # kill command
    
    try:
        rclpy.spin(server)
    except KeyboardInterrupt:
        cprint("Keyboard interrupt received, shutting down...", "yellow")
        server._cleanup()
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
