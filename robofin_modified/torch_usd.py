import os
import torch
import numpy as np
from collections import OrderedDict
from pxr import Usd, UsdGeom, Sdf
from typing import Dict, List, Optional, Tuple, Union, Any


class TorchUSD:
    """
    A PyTorch-based implementation for USD robot parsing and forward kinematics.
    This class loads robot data from a USD file and provides batched forward kinematics.
    """

    def __init__(self, name: str, device: Optional[torch.device] = None):
        """
        Initialize the TorchUSD class.

        Parameters
        ----------
        name : str
            The name of the robot.
        device : torch.device, optional
            The device to use for PyTorch tensors.
        """
        self.name = name
        self.device = device if device is not None else torch.device("cpu")
        self.links = []
        self.joints = []
        self._reverse_topo = []
        self._paths_to_base = {}
        self._G = None
        self._joint_map = {}
        self.actuated_joints = []

    @staticmethod
    def load(
        file_path: str,
        robot_prim_path: str = "/robot",
        device: Optional[torch.device] = None,
    ) -> "TorchUSD":
        """
        Load a robot from a USD file.

        Parameters
        ----------
        file_path : str
            Path to the USD file.
        robot_prim_path : str, optional
            Path to the robot prim in the USD file.
        device : torch.device, optional
            The device to use for PyTorch tensors.

        Returns
        -------
        TorchUSD
            A TorchUSD instance representing the robot.
        """
        # Open the USD stage
        stage = Usd.Stage.Open(file_path)
        if not stage:
            raise ValueError(f"Failed to open USD file: {file_path}")

        # Get the robot prim
        robot_prim = stage.GetPrimAtPath(Sdf.Path(robot_prim_path))
        if not robot_prim:
            raise ValueError(f"Robot prim not found at path: {robot_prim_path}")

        # Create a new TorchUSD instance
        robot_name = robot_prim.GetName()
        robot = TorchUSD(robot_name, device)

        # Parse the robot data
        robot._parse_robot_data(stage, robot_prim)

        return robot

    def _parse_robot_data(self, stage: Usd.Stage, robot_prim: Usd.Prim):
        """
        Parse robot data from a USD prim.

        Parameters
        ----------
        stage : Usd.Stage
            The USD stage.
        robot_prim : Usd.Prim
            The robot prim.
        """
        # Create a directed graph for the kinematic chain
        import networkx as nx

        self._G = nx.DiGraph()

        # Get the xform cache for transforms
        xform_cache = UsdGeom.XformCache()

        # Find the root joint (fixed joint with empty body0)
        root_joint = self._find_root_joint(robot_prim)
        if not root_joint:
            raise ValueError("Root joint not found in robot prim")

        # Get the first link from the root joint
        root_link_path = self._get_joint_body1(root_joint)
        if not root_link_path:
            raise ValueError("Root link not found in root joint")

        # Add the root link to the graph
        self._G.add_node(root_link_path)

        # Find all joints and links
        self._find_joints_and_links(robot_prim, root_link_path, stage, xform_cache)

        # Build the kinematic chain
        self._build_kinematic_chain(root_link_path)

    def _find_root_joint(self, robot_prim: Usd.Prim) -> Optional[Usd.Prim]:
        """
        Find the root joint (fixed joint with empty body0) in the robot prim.

        Parameters
        ----------
        robot_prim : Usd.Prim
            The robot prim.

        Returns
        -------
        Usd.Prim, optional
            The root joint prim, or None if not found.
        """
        # Traverse the prim hierarchy to find the root joint
        for prim in robot_prim.GetChildren():
            if prim.GetTypeName() == "PhysicsFixedJoint":
                # Check if body0 is empty
                body0_rel = prim.GetAttribute("body0")
                if body0_rel and not body0_rel.Get():
                    return prim
            # Recursively search children
            result = self._find_root_joint(prim)
            if result:
                return result
        return None

    def _get_joint_body1(self, joint: Usd.Prim) -> Optional[str]:
        """
        Get the body1 target of a joint.

        Parameters
        ----------
        joint : Usd.Prim
            The joint prim.

        Returns
        -------
        str, optional
            The body1 target path, or None if not found.
        """
        body1_rel = joint.GetAttribute("body1")
        if body1_rel:
            targets = body1_rel.Get()
            if targets and len(targets) > 0:
                return targets[0]
        return None

    def _find_joints_and_links(
        self,
        robot_prim: Usd.Prim,
        root_link_path: str,
        stage: Usd.Stage,
        xform_cache: UsdGeom.XformCache,
    ):
        """
        Find all joints and links in the robot prim.

        Parameters
        ----------
        robot_prim : Usd.Prim
            The robot prim.
        root_link_path : str
            The path to the root link.
        stage : Usd.Stage
            The USD stage.
        xform_cache : UsdGeom.XformCache
            The xform cache for transforms.
        """

        # Create a class to represent a link
        class Link:
            def __init__(self, name, path, transform):
                self.name = name
                self.path = path
                self.transform = transform
                self.visuals = []

        # Create a class to represent a joint
        class Joint:
            def __init__(
                self, name, joint_type, parent, child, axis, transform, limits=None
            ):
                self.name = name
                self.joint_type = joint_type
                self.parent = parent
                self.child = child
                self.axis = axis
                self.transform = transform
                self.limits = limits
                self.mimic = None

        # Find all joints and links
        joints = []
        links = {}

        # Add the root link
        root_link_prim = stage.GetPrimAtPath(Sdf.Path(root_link_path))
        root_link = Link(
            root_link_prim.GetName(),
            root_link_path,
            self._get_transform(root_link_prim, xform_cache),
        )
        links[root_link_path] = root_link

        # Find all joints and links
        self._traverse_kinematic_chain(
            robot_prim, root_link_path, stage, xform_cache, joints, links
        )

        # Store the joints and links
        self.joints = joints
        self.links = list(links.values())

        # Create a map from joint names to joints
        self._joint_map = {joint.name: joint for joint in joints}

        # Identify actuated joints
        self.actuated_joints = [
            joint for joint in joints if joint.joint_type in ["revolute", "prismatic"]
        ]

    def _traverse_kinematic_chain(
        self,
        robot_prim: Usd.Prim,
        current_link_path: str,
        stage: Usd.Stage,
        xform_cache: UsdGeom.XformCache,
        joints: List,
        links: Dict,
    ):
        """
        Traverse the kinematic chain to find all joints and links.

        Parameters
        ----------
        robot_prim : Usd.Prim
            The robot prim.
        current_link_path : str
            The path to the current link.
        stage : Usd.Stage
            The USD stage.
        xform_cache : UsdGeom.XformCache
            The xform cache for transforms.
        joints : List
            The list of joints.
        links : Dict
            The dictionary of links.
        """
        # Find all joints that have the current link as body0
        for prim in robot_prim.GetChildren():
            if prim.GetTypeName().startswith("Physics") and prim.GetTypeName().endswith(
                "Joint"
            ):
                body0_rel = prim.GetAttribute("body0")
                if body0_rel:
                    targets = body0_rel.Get()
                    if targets and len(targets) > 0 and targets[0] == current_link_path:
                        # This joint has the current link as body0
                        body1_path = self._get_joint_body1(prim)
                        if body1_path:
                            # Get the child link
                            child_link_prim = stage.GetPrimAtPath(Sdf.Path(body1_path))
                            if child_link_prim:
                                # Create the child link
                                child_link = Link(
                                    child_link_prim.GetName(),
                                    body1_path,
                                    self._get_transform(child_link_prim, xform_cache),
                                )
                                links[body1_path] = child_link

                                # Add the link to the graph
                                self._G.add_node(body1_path)
                                self._G.add_edge(
                                    current_link_path, body1_path, joint=prim
                                )

                                # Create the joint
                                joint_type = (
                                    prim.GetTypeName().replace("Physics", "").lower()
                                )
                                axis = self._get_joint_axis(prim)
                                transform = self._get_transform(prim, xform_cache)
                                limits = self._get_joint_limits(prim)

                                joint = Joint(
                                    prim.GetName(),
                                    joint_type,
                                    current_link_path,
                                    body1_path,
                                    axis,
                                    transform,
                                    limits,
                                )
                                joints.append(joint)

                                # Recursively traverse the kinematic chain
                                self._traverse_kinematic_chain(
                                    robot_prim,
                                    body1_path,
                                    stage,
                                    xform_cache,
                                    joints,
                                    links,
                                )

            # Recursively search children
            self._traverse_kinematic_chain(
                prim, current_link_path, stage, xform_cache, joints, links
            )

    def _get_transform(
        self, prim: Usd.Prim, xform_cache: UsdGeom.XformCache
    ) -> torch.Tensor:
        """
        Get the transform of a prim.

        Parameters
        ----------
        prim : Usd.Prim
            The prim.
        xform_cache : UsdGeom.XformCache
            The xform cache for transforms.

        Returns
        -------
        torch.Tensor
            The transform matrix.
        """
        xformable = UsdGeom.Xformable(prim)
        if xformable:
            transform = xform_cache.GetLocalToWorldTransform(prim)
            # Convert to torch tensor and transpose (USD uses row-major, PyTorch uses column-major)
            return torch.tensor(transform.GetArray(), device=self.device).transpose(
                0, 1
            )
        return torch.eye(4, device=self.device)

    def _get_joint_axis(self, joint: Usd.Prim) -> torch.Tensor:
        """
        Get the axis of a joint.

        Parameters
        ----------
        joint : Usd.Prim
            The joint prim.

        Returns
        -------
        torch.Tensor
            The joint axis.
        """
        axis_attr = joint.GetAttribute("axis")
        if axis_attr:
            axis = axis_attr.Get()
            if axis:
                # Normalize the axis
                axis = np.array(axis)
                axis = axis / np.linalg.norm(axis)
                return torch.tensor(axis, device=self.device)
        # Default axis is [1, 0, 0]
        return torch.tensor([1.0, 0.0, 0.0], device=self.device)

    def _get_joint_limits(self, joint: Usd.Prim) -> Optional[Tuple[float, float]]:
        """
        Get the limits of a joint.

        Parameters
        ----------
        joint : Usd.Prim
            The joint prim.

        Returns
        -------
        Tuple[float, float], optional
            The joint limits (lower, upper), or None if not found.
        """
        lower_attr = joint.GetAttribute("lower")
        upper_attr = joint.GetAttribute("upper")
        if lower_attr and upper_attr:
            lower = lower_attr.Get()
            upper = upper_attr.Get()
            if lower is not None and upper is not None:
                return (lower, upper)
        return None

    def _build_kinematic_chain(self, root_link_path: str):
        """
        Build the kinematic chain from the root link.

        Parameters
        ----------
        root_link_path : str
            The path to the root link.
        """
        # Compute the reverse topological order
        self._reverse_topo = list(nx.dfs_postorder_nodes(self._G, root_link_path))

        # Compute paths to base for each link
        for link_path in self._G.nodes():
            path = list(nx.shortest_path(self._G, link_path, root_link_path))
            self._paths_to_base[link_path] = path

    def _process_cfgs(
        self, cfgs: Union[torch.Tensor, Dict, List[Dict]]
    ) -> Tuple[Dict, int]:
        """
        Process joint configurations into a dictionary mapping joints to configuration values.

        Parameters
        ----------
        cfgs : torch.Tensor, Dict, or List[Dict]
            The joint configurations.

        Returns
        -------
        Tuple[Dict, int]
            A tuple containing (joint_cfg, n_cfgs).
        """
        joint_cfg = {}

        if isinstance(cfgs, torch.Tensor):
            # If cfgs is a tensor, assume it's a batch of configurations
            n_cfgs = cfgs.shape[0]
            for i, joint in enumerate(self.actuated_joints):
                joint_cfg[joint] = cfgs[:, i]
        elif isinstance(cfgs, dict):
            # If cfgs is a dictionary, assume it's a single configuration
            n_cfgs = 1
            for joint_name, value in cfgs.items():
                if isinstance(joint_name, str):
                    # If joint_name is a string, find the joint
                    for joint in self.joints:
                        if joint.name == joint_name:
                            joint_cfg[joint] = torch.tensor([value], device=self.device)
                            break
                else:
                    # If joint_name is a joint, use it directly
                    joint_cfg[joint_name] = torch.tensor([value], device=self.device)
        elif isinstance(cfgs, list):
            # If cfgs is a list, assume it's a list of configurations
            n_cfgs = len(cfgs)
            for i, joint in enumerate(self.actuated_joints):
                joint_cfg[joint] = torch.tensor(
                    [cfg[joint.name] for cfg in cfgs], device=self.device
                )
        else:
            # If cfgs is None, use zero configurations
            n_cfgs = 1
            for joint in self.actuated_joints:
                joint_cfg[joint] = torch.zeros(1, device=self.device)

        return joint_cfg, n_cfgs

    def _rotation_matrices(
        self, angles: torch.Tensor, axis: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute rotation matrices from angle/axis representations.

        Parameters
        ----------
        angles : torch.Tensor
            The angles.
        axis : torch.Tensor
            The axis.

        Returns
        -------
        torch.Tensor
            The rotation matrices.
        """
        # Normalize the axis
        axis = axis / torch.norm(axis)

        # Compute sin and cos of angles
        sina = torch.sin(angles)
        cosa = torch.cos(angles)

        # Create identity matrices
        M = torch.eye(4, device=self.device).repeat((len(angles), 1, 1))

        # Set diagonal elements
        M[:, 0, 0] = cosa
        M[:, 1, 1] = cosa
        M[:, 2, 2] = cosa

        # Add outer product terms
        M[:, :3, :3] += (
            torch.ger(axis, axis).repeat((len(angles), 1, 1))
            * (1.0 - cosa)[:, None, None]
        )

        # Add cross product terms
        M[:, :3, :3] += (
            torch.tensor(
                [
                    [0.0, -axis[2], axis[1]],
                    [axis[2], 0.0, -axis[0]],
                    [-axis[1], axis[0], 0.0],
                ],
                device=self.device,
            ).repeat((len(angles), 1, 1))
            * sina[:, None, None]
        )

        return M

    def get_child_poses(
        self, joint: Any, cfg: Optional[torch.Tensor], n_cfgs: int
    ) -> torch.Tensor:
        """
        Get the poses of a joint's child link relative to its parent link.

        Parameters
        ----------
        joint : Any
            The joint.
        cfg : torch.Tensor, optional
            The joint configuration.
        n_cfgs : int
            The number of configurations.

        Returns
        -------
        torch.Tensor
            The child poses.
        """
        if cfg is None:
            # If cfg is None, return the joint's transform
            return joint.transform.repeat((n_cfgs, 1, 1))

        if joint.joint_type == "fixed":
            # If the joint is fixed, return the joint's transform
            return joint.transform.repeat((n_cfgs, 1, 1))

        elif joint.joint_type in ["revolute", "continuous"]:
            # If the joint is revolute or continuous, compute the rotation matrix
            if cfg is None:
                cfg = torch.zeros(n_cfgs, device=self.device)

            # Compute the rotation matrix
            rot_mat = self._rotation_matrices(cfg, joint.axis)

            # Multiply the rotation matrix by the joint's transform
            return torch.matmul(joint.transform, rot_mat)

        elif joint.joint_type == "prismatic":
            # If the joint is prismatic, compute the translation matrix
            if cfg is None:
                cfg = torch.zeros(n_cfgs, device=self.device)

            # Create translation matrices
            trans_mat = torch.eye(4, device=self.device).repeat((n_cfgs, 1, 1))
            trans_mat[:, :3, 3] = joint.axis * cfg[:, None]

            # Multiply the translation matrix by the joint's transform
            return torch.matmul(joint.transform, trans_mat)

        else:
            # If the joint type is not supported, raise an error
            raise ValueError(f"Unsupported joint type: {joint.joint_type}")

    def link_fk_batch(
        self,
        cfgs: Optional[Union[torch.Tensor, Dict, List[Dict]]] = None,
        use_names: bool = False,
    ) -> Dict:
        """
        Compute the poses of the robot's links via forward kinematics in a batch.

        Parameters
        ----------
        cfgs : torch.Tensor, Dict, or List[Dict], optional
            The joint configurations.
        use_names : bool, optional
            If True, the returned dictionary will have keys that are string link names.

        Returns
        -------
        Dict
            A map from links to a (n,4,4) vector of homogenous transform matrices.
        """
        # Process the configurations
        joint_cfgs, n_cfgs = self._process_cfgs(cfgs)

        # Compute FK mapping each link to a vector of matrices, one matrix per cfg
        fk = OrderedDict()

        for link_path in self._reverse_topo:
            # Initialize the poses for this link
            poses = torch.eye(4, device=self.device).repeat((n_cfgs, 1, 1))

            # Get the path from this link to the base
            path = self._paths_to_base[link_path]

            # Traverse the path from this link to the base
            for i in range(len(path) - 1):
                child = path[i]
                parent = path[i + 1]

                # Get the joint connecting the child and parent
                joint_data = self._G.get_edge_data(child, parent)
                if not joint_data or "joint" not in joint_data:
                    continue

                joint_prim = joint_data["joint"]

                # Find the corresponding joint object
                joint = None
                for j in self.joints:
                    if j.parent == parent and j.child == child:
                        joint = j
                        break

                if not joint:
                    continue

                # Get the joint configuration
                cfg_vals = None
                if joint.mimic is not None:
                    # If the joint is a mimic joint, get the configuration from the mimicked joint
                    mimic_joint = self._joint_map[joint.mimic.joint]
                    if mimic_joint in joint_cfgs:
                        cfg_vals = joint_cfgs[mimic_joint]
                        cfg_vals = (
                            joint.mimic.multiplier * cfg_vals + joint.mimic.offset
                        )
                elif joint in joint_cfgs:
                    # If the joint is in the configuration, use its value
                    cfg_vals = joint_cfgs[joint]

                # Get the child poses
                child_poses = self.get_child_poses(joint, cfg_vals, n_cfgs)

                # Update the poses
                poses = torch.matmul(child_poses, poses.type_as(child_poses))

                # If the parent is already in the FK dictionary, multiply by its transform
                if parent in fk:
                    poses = torch.matmul(fk[parent], poses.type_as(fk[parent]))
                    break

            # Store the poses for this link
            fk[link_path] = poses

        # If use_names is True, convert the keys to link names
        if use_names:
            return {link.name: fk[link.path] for link in self.links if link.path in fk}

        return fk

    def visual_geometry_fk_batch(
        self, cfgs: Optional[Union[torch.Tensor, Dict, List[Dict]]] = None
    ) -> Dict:
        """
        Compute the poses of the robot's visual geometries using forward kinematics.

        Parameters
        ----------
        cfgs : torch.Tensor, Dict, or List[Dict], optional
            The joint configurations.

        Returns
        -------
        Dict
            A map from visual geometries to their poses.
        """
        # Get the link poses
        lfk = self.link_fk_batch(cfgs=cfgs)

        # Initialize the FK dictionary
        fk = OrderedDict()

        # For each link, compute the poses of its visual geometries
        for link in self.links:
            if link.path in lfk:
                # For each visual in the link, compute its pose
                for visual in link.visuals:
                    # Multiply the link pose by the visual's transform
                    fk[visual] = torch.matmul(
                        lfk[link.path], visual.transform.type_as(lfk[link.path])
                    )

        return fk
