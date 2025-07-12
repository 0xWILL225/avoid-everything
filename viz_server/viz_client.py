"""
Thin helper that hides the ZeroMQ details; no ROS imports needed.

Example
-------
import numpy as np, viz_client as V
V.connect("ur10.urdf")
V.publish_robot_points(np.random.rand(2_048, 3).astype("f4"))
V.publish_target_points(np.random.rand(1_024, 3).astype("f4"))
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import torch as T
import yaml
import zmq
from termcolor import cprint
import os

PORT        = 5556
SERVER_CMD  = "source /opt/ros/humble/setup.bash && python3 -c \"import sys; sys.path.insert(0, '/workspace/viz_server/src'); from viz_server.server import main; main()\""

_ctx        = zmq.Context.instance()
_sock: zmq.Socket[Any] | None = None
_connected  = False
_base_link_name: str = "base_link"  # Default, will be loaded from config


# ====================================================================== #
# Server bootstrap helpers
# ====================================================================== #
def _server_alive() -> bool:
    """Return True iff a viz_server REP socket is already up."""
    try:
        s = _ctx.socket(zmq.REQ)
        s.setsockopt(zmq.LINGER, 0)
        s.setsockopt(zmq.RCVTIMEO, 1000)  # 1 second timeout
        s.connect(f"tcp://127.0.0.1:{PORT}")
        s.send_json({"cmd": "ping"})
        s.recv_json(flags=0)
        s.close(0)
        return True
    except Exception:   # noqa: BLE001
        return False


def connect(urdf: str, *, port: int = 5556) -> None:
    """
    Ensure viz_server is running and obtain a REQ socket.

    Parameters
    ----------
    urdf : str
        Path to the robot URDF (only needed on first call).
    port : int, default 5556
        ZeroMQ port to connect to.
    """
    global _sock, _connected, PORT, _base_link_name
    PORT = port

    if not _server_alive():
        cprint("viz_server not running — starting…", "yellow")
        if not Path(urdf).is_file():
            raise FileNotFoundError(urdf)
        
        cmd = f"{SERVER_CMD} --urdf {urdf}"
        
        subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            shell=True,
            executable="/bin/bash"
        )
        
        # Wait up to 10 seconds for server to start
        for i in range(100):  # 10 seconds with 0.1s sleep
            if _server_alive():
                break
            time.sleep(0.1)
        
        if not _server_alive():
            raise RuntimeError("Server failed to start within 10 seconds")

    # Load base link name from config
    _base_link_name = _load_base_link_name(urdf)
    
    _sock = _ctx.socket(zmq.REQ)
    assert _sock is not None
    _sock.connect(f"tcp://127.0.0.1:{PORT}")
    _connected = True; cprint("Connected to viz_server", "green")


# ====================================================================== #
# Low-level send helper
# ====================================================================== #
def _send(hdr: dict, payload: bytes | None = None) -> None:
    if not _connected or _sock is None:
        raise RuntimeError("Call viz_client.connect() first")
    if payload is None:
        _sock.send_json(hdr)
    else:
        _sock.send_json(hdr, zmq.SNDMORE)
        _sock.send(payload, copy=False)
    resp = _sock.recv_json()
    if not isinstance(resp, dict) or resp.get("status") != "ok":
        msg = resp.get("msg", "unknown error") if isinstance(resp, dict) else str(resp)
        cprint(f"Server error: {msg}", "red")


# ====================================================================== #
# Helper functions
# ====================================================================== #
def _load_base_link_name(urdf_path: str) -> str:
    """Load base link name from link_config.yaml in the same directory as URDF."""
    try:
        urdf_dir = Path(urdf_path).parent
        config_path = urdf_dir / "link_config.yaml"
        
        if not config_path.exists():
            cprint(f"Warning: {config_path} not found, using default 'base_link'", "yellow")
            return "base_link"
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        base_link = config.get("link_config", {}).get("base_link_name", "base_link")
        return base_link
        
    except Exception as e:
        cprint(f"Error loading link_config.yaml: {e}, using default 'base_link'", "yellow")
        return "base_link"


# ====================================================================== #
# Public API
# ====================================================================== #
def publish_robot_pointcloud(
    points: np.ndarray | T.Tensor,
    *,
    frame: str | None = None,
    name: str = "robot_cloud"
) -> None:
    """
    Publish robot point cloud.

    Parameters
    ----------
    points : (N, 3) NumPy array or Torch tensor
    frame  : TF frame name (default: base link from config)
    name   : logical name of the cloud
    """
    if frame is None:
        frame = _base_link_name
    
    arr = points.cpu().numpy() if isinstance(points, T.Tensor) else points
    arr = arr.astype(np.float32)
    hdr = {
        "cmd":"pointcloud", "frame":frame, "pc_type":"robot_points",
        "dtype":str(arr.dtype), "shape":arr.shape,
        "name":name
    }
    _send(hdr, arr.tobytes())


def publish_target_pointcloud(
    points: np.ndarray | T.Tensor,
    *,
    frame: str | None = None,
    name: str = "target_cloud"
) -> None:
    """
    Publish target point cloud.

    Parameters
    ----------
    points : (N, 3) NumPy array or Torch tensor
    frame  : TF frame name (default: base link from config)  
    name   : logical name of the cloud
    """
    if frame is None:
        frame = _base_link_name
    
    arr = points.cpu().numpy() if isinstance(points, T.Tensor) else points
    arr = arr.astype(np.float32)
    hdr = {
        "cmd":"pointcloud", "frame":frame, "pc_type":"target_points",
        "dtype":str(arr.dtype), "shape":arr.shape,
        "name":name
    }
    _send(hdr, arr.tobytes())


def publish_obstacle_pointcloud(
    points: np.ndarray | T.Tensor,
    *,
    frame: str | None = None,
    name: str = "obstacle_cloud"
) -> None:
    """
    Publish obstacle point cloud.

    Parameters
    ----------
    points : (N, 3) NumPy array or Torch tensor
    frame  : TF frame name (default: base link from config)
    name   : logical name of the cloud
    """
    if frame is None:
        frame = _base_link_name
    
    arr = points.cpu().numpy() if isinstance(points, T.Tensor) else points
    arr = arr.astype(np.float32)
    hdr = {
        "cmd":"pointcloud", "frame":frame, "pc_type":"obstacle_points",
        "dtype":str(arr.dtype), "shape":arr.shape,
        "name":name
    }
    _send(hdr, arr.tobytes())


def clear_robot_pointcloud(*, frame: str | None = None, name: str = "robot_cloud") -> None:
    """Clear robot point cloud by publishing empty cloud."""
    if frame is None:
        frame = _base_link_name
    empty_points = np.array([], dtype=np.float32).reshape(0, 3)
    hdr = {
        "cmd":"pointcloud", "frame":frame, "pc_type":"robot_points",
        "dtype":str(empty_points.dtype), "shape":empty_points.shape,
        "name":name
    }
    _send(hdr, empty_points.tobytes())


def clear_target_pointcloud(*, frame: str | None = None, name: str = "target_cloud") -> None:
    """Clear target point cloud by publishing empty cloud."""
    if frame is None:
        frame = _base_link_name
    empty_points = np.array([], dtype=np.float32).reshape(0, 3)
    hdr = {
        "cmd":"pointcloud", "frame":frame, "pc_type":"target_points",
        "dtype":str(empty_points.dtype), "shape":empty_points.shape,
        "name":name
    }
    _send(hdr, empty_points.tobytes())


def clear_obstacle_pointcloud(*, frame: str | None = None, name: str = "obstacle_cloud") -> None:
    """Clear obstacle point cloud by publishing empty cloud."""
    if frame is None:
        frame = _base_link_name
    empty_points = np.array([], dtype=np.float32).reshape(0, 3)
    hdr = {
        "cmd":"pointcloud", "frame":frame, "pc_type":"obstacle_points",
        "dtype":str(empty_points.dtype), "shape":empty_points.shape,
        "name":name
    }
    _send(hdr, empty_points.tobytes())


# Legacy functions for backward compatibility
def publish_pointcloud(
    points: np.ndarray | T.Tensor,
    *,
    frame: str | None = None,
    pc_type: str = "robot_points",
    name: str = "cloud"
) -> None:
    """
    Stream an XYZ point cloud to the server.
    
    DEPRECATED: Use publish_robot_pointcloud(), publish_target_pointcloud(), 
    or publish_obstacle_pointcloud() instead.

    Parameters
    ----------
    points : (N, 3) NumPy array or Torch tensor
    frame  : TF frame name (default: base link from config)
    pc_type : point cloud type - 'robot_points', 'target_points', or 'obstacle_points'
    name   : logical name of the cloud (colors set in RViz config)
    """
    # Use base link as default frame
    if frame is None:
        frame = _base_link_name
    
    # Validate pc_type
    valid_types = ["robot_points", "target_points", "obstacle_points"]
    if pc_type not in valid_types:
        raise ValueError(f"pc_type must be one of {valid_types}, got '{pc_type}'")
    
    arr = points.cpu().numpy() if isinstance(points, T.Tensor) else points
    arr = arr.astype(np.float32)  # Ensure consistent dtype
    hdr = {
        "cmd":"pointcloud", "frame":frame, "pc_type":pc_type,
        "dtype":str(arr.dtype), "shape":arr.shape,
        "name":name
    }
    _send(hdr, arr.tobytes())


def publish_robot_points(
    points: np.ndarray | T.Tensor,
    *,
    frame: str | None = None,
    name: str = "robot_cloud"
) -> None:
    """DEPRECATED: Use publish_robot_pointcloud() instead."""
    publish_robot_pointcloud(points, frame=frame, name=name)


def publish_target_points(
    points: np.ndarray | T.Tensor,
    *,
    frame: str | None = None,
    name: str = "target_cloud"
) -> None:
    """DEPRECATED: Use publish_target_pointcloud() instead."""
    publish_target_pointcloud(points, frame=frame, name=name)


def publish_obstacle_points(
    points: np.ndarray | T.Tensor,
    *,
    frame: str | None = None,
    name: str = "obstacle_cloud"
) -> None:
    """DEPRECATED: Use publish_obstacle_pointcloud() instead."""
    publish_obstacle_pointcloud(points, frame=frame, name=name)


def publish_joints(joints: Dict[str, float]) -> None:
    """
    Show the robot in a single configuration.

    Parameters
    ----------
    joints : Dict[str, float]
        Joint angles in radians. Must include all movable joints OR parent joints
        of mimic relationships. Mimic joints (like panda_finger_joint2) are 
        automatically computed from their parent joints (like panda_finger_joint1).
        
    Examples
    --------
    # Traditional: set all joints including mimics
    publish_joints({"joint1": 0.0, "finger_joint1": 0.02, "finger_joint2": 0.02})
    
    # Better: only set parent joints, mimics computed automatically  
    publish_joints({"joint1": 0.0, "finger_joint1": 0.02})
    """
    _send({"cmd":"joints", "joints":joints})


def publish_trajectory(
    waypoints: List[Dict[str, float]],
    *,
    segment_duration: float = 1.0,
    rate_hz: float = 30.0
) -> None:
    """
    Animate a list of joint dictionaries.

    Parameters
    ----------
    waypoints : ordered list of full joint maps
    segment_duration : seconds spent interpolating between each pair
    rate_hz : animation framerate in Hz (higher = smoother)
    """
    _send({"cmd":"trajectory",
           "waypoints":waypoints,
           "segment_duration":segment_duration,
           "rate_hz":rate_hz})


def publish_ghost_end_effector(
    pose: List[float],      # [x y z qx qy qz qw]
    *,
    color: List[float] | None = None,
    scale: float = 1.0,
    alpha: float = 0.5
) -> None:
    """
    Display a translucent mesh of the entire end effector at an arbitrary pose.
    
    This shows the end effector base link plus all visual links (e.g., fingers)
    with proper forward kinematics applied based on current joint states.
    
    Parameters
    ----------
    pose : List[float]
        [x, y, z, qx, qy, qz, qw] position and orientation for end effector base
    color : List[float] | None
        [r, g, b] color values in 0-1 range (default green)
    scale : float
        Scale factor for the mesh
    alpha : float
        Alpha/transparency value in 0-1 range (0=transparent, 1=opaque)
    """
    if color is None: color = [0, 1, 0]  # Default green color
    _send({"cmd":"ghost_end_effector", "pose":pose, "color":color, "scale":scale, "alpha":alpha})


def publish_ghost_robot(
    configuration: Dict[str, float],
    *,
    color: List[float] | None = None,
    scale: float = 1.0,
    alpha: float = 0.5
) -> None:
    """
    Display a translucent mesh of the entire robot at an arbitrary configuration.
    
    This shows all robot links with visual geometry using forward kinematics
    to compute the pose of each link based on the given joint configuration.
    
    Parameters
    ----------
    configuration : Dict[str, float]
        Joint angles in radians for all robot joints. Must include all movable joints
        OR parent joints of mimic relationships. Mimic joints are automatically computed.
    color : List[float] | None
        [r, g, b] color values in 0-1 range (default green)
    scale : float
        Scale factor for all meshes
    alpha : float
        Alpha/transparency value in 0-1 range (0=transparent, 1=opaque)
        
    Examples
    --------
    # Show robot in a specific configuration
    config = {
        "joint1": 0.0,
        "joint2": -0.785,
        "joint3": 0.0,
        "joint4": -2.356,
        "joint5": 0.0,
        "joint6": 1.571,
        "joint7": 0.785,
        "finger_joint1": 0.02,
    }
    publish_ghost_robot(config, color=[1, 0, 0], alpha=0.3)
    """
    if color is None: color = [0, 1, 0]  # Default green color
    _send({"cmd":"ghost_robot", "configuration":configuration, "color":color, "scale":scale, "alpha":alpha})


def clear_ghost_end_effector() -> None:
    """
    Clear all ghost end effector markers from RViz.
    
    This removes all translucent end effector meshes that were previously
    published with publish_ghost_end_effector().
    """
    _send({"cmd":"clear_ghost_end_effector"})


def clear_ghost_robot() -> None:
    """
    Clear all ghost robot markers from RViz.
    
    This removes all translucent robot meshes that were previously
    published with publish_ghost_robot().
    """
    _send({"cmd":"clear_ghost_robot"})


def shutdown() -> None:
    """
    Shutdown the viz_server and its robot_state_publisher.
    
    This will cleanly terminate the viz_server process and its
    robot_state_publisher subprocess.
    """
    global _connected
    
    if not _connected or _sock is None:
        cprint("Not connected to viz_server", "yellow")
        return
    
    try:
        _send({"cmd":"shutdown"})
        cprint("Sent shutdown command to viz_server", "green")
    except Exception as e:
        cprint(f"Error sending shutdown command: {e}", "red")
    
    # Reset connection state
    _connected = False
