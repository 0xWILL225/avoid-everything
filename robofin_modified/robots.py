from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from geometrout.transform import SE3, SO3
import torch

# from ikfast_franka_panda import get_fk, get_ik


class Robot:
    """
    A generic robot class that dynamically reads information from a URDF file.
    This class can be used for any robot manipulator, replacing the need for
    specialized robot classes.
    """

    def __init__(
        self,
        urdf_path: str,
        neutral_config: np.ndarray = None,
        end_effector_name: str = None,
    ):
        """
        Initialize the Robot with a URDF file path.

        Args:
            urdf_path: Path to the URDF file. If None, uses the default path.
        """
        if urdf_path is None:
            raise ValueError("urdf_path is required")

        self.urdf_path = urdf_path

        # Parse the URDF file
        self.tree = ET.parse(self.urdf_path)
        self.root = self.tree.getroot()

        # Extract robot name
        self.name = self.root.get("name")

        # Extract joint information
        self._extract_joint_info()

        # Extract link information
        self._extract_link_info()

        # Set default end effector
        if end_effector_name is None:
            raise ValueError("end_effector_name is required")

        self.default_eff = end_effector_name

        # Define end effector transformations
        self._define_eff_transforms()

        # Define a neutral configuration (all joints at 0)
        if neutral_config is None:
            self.NEUTRAL = np.zeros(self.DOF)
        else:
            self.NEUTRAL = neutral_config

        # Define collision spheres (placeholder - can be overridden)
        self.SPHERES = []

    def _extract_joint_info(self):
        """Extract joint information from the URDF."""
        # Find all revolute joints
        self.joints = []
        self.joint_names = []
        self.joint_limits = []
        self.velocity_limits = []

        for joint in self.root.findall(".//joint"):
            joint_type = joint.get("type")
            joint_name = joint.get("name")

            if joint_type == "revolute":
                self.joints.append(joint)
                self.joint_names.append(joint_name)

                # Extract joint limits
                limit = joint.find("limit")
                if limit is not None:
                    lower = float(limit.get("lower"))
                    upper = float(limit.get("upper"))
                    velocity = float(limit.get("velocity", 0.0))

                    self.joint_limits.append((lower, upper))
                    self.velocity_limits.append(velocity)

        # Convert to numpy arrays
        self.JOINT_LIMITS = np.array(self.joint_limits)
        self.VELOCITY_LIMIT = np.array(self.velocity_limits)

        # Set DOF based on number of revolute joints
        self.DOF = len(self.joints)

    def _extract_link_info(self):
        """Extract link information from the URDF."""
        self.links = []
        self.link_names = []

        for link in self.root.findall(".//link"):
            link_name = link.get("name")
            self.links.append(link)
            self.link_names.append(link_name)

    def _define_eff_transforms(self):
        """Define end effector transformations."""
        # Initialize with the default end effector
        self.EFF_LIST = set([self.default_eff])
        self.EFF_T_LIST = {}

    def within_limits(self, config: np.ndarray) -> bool:
        """Check if a configuration is within joint limits."""
        # We have to add a small buffer because of float math
        return np.all(config >= self.JOINT_LIMITS[:, 0] - 1e-5) and np.all(
            config <= self.JOINT_LIMITS[:, 1] + 1e-5
        )

    def normalize_joints(self, config, limits=(-1, 1)):
        """
        Normalizes joint angles to be within a specified range according to the robot's
        joint limits.

        Args:
            config: Joint configuration as numpy array or torch.Tensor. Can have dims
                  [DOF] if a single configuration
                  [B, DOF] if a batch of configurations
                  [B, T, DOF] if a batched time-series of configurations
            limits: Tuple of (min, max) values to normalize to, default (-1, 1)

        Returns:
            Normalized joint angles with same shape and type as input
        """
        # Handle torch tensors differently than numpy arrays
        if hasattr(config, "device") and hasattr(
            config, "dtype"
        ):  # It's a torch tensor

            assert isinstance(config, torch.Tensor), "Expected torch.Tensor"

            # Check input dimensions match the robot's DOF
            input_dof_dim = 0 if config.dim() == 1 else (1 if config.dim() == 2 else 2)
            input_dof = config.size(input_dof_dim)
            assert input_dof == self.DOF, f"Expected {self.DOF} DOF, got {input_dof}"

            # Get joint limits as a tensor
            joint_limits = torch.tensor(
                self.JOINT_LIMITS, dtype=config.dtype, device=config.device
            )

            # Calculate normalization for the configuration
            joint_range = joint_limits[:, 1] - joint_limits[:, 0]
            joint_min = joint_limits[:, 0]

            # Reshape joint limits if needed for broadcasting
            if config.dim() > 1:
                for _ in range(config.dim() - 1):
                    joint_range = joint_range.unsqueeze(0)
                    joint_min = joint_min.unsqueeze(0)

            # Normalize: first to [0,1], then to the target range
            normalized = (config - joint_min) / joint_range
            normalized = normalized * (limits[1] - limits[0]) + limits[0]

            return normalized

        else:  # It's a numpy array or list
            # Ensure the config is a numpy array
            if not isinstance(config, np.ndarray):
                config = np.array(config)

            # Check input dimensions match the robot's DOF
            input_dof_dim = 0 if config.ndim == 1 else (1 if config.ndim == 2 else 2)
            input_dof = config.shape[input_dof_dim]
            assert input_dof == self.DOF, f"Expected {self.DOF} DOF, got {input_dof}"

            # Get joint limits
            joint_limits = self.JOINT_LIMITS

            # Calculate normalization for the configuration
            joint_range = joint_limits[:, 1] - joint_limits[:, 0]
            joint_min = joint_limits[:, 0]

            # Reshape joint limits if needed for broadcasting
            if config.ndim > 1:
                for _ in range(config.ndim - 1):
                    joint_range = joint_range[np.newaxis, ...]
                    joint_min = joint_min[np.newaxis, ...]

            # Normalize: first to [0,1], then to the target range
            normalized = (config - joint_min) / joint_range
            normalized = normalized * (limits[1] - limits[0]) + limits[0]

            return normalized

    def unnormalize_joints(self, config, limits=(-1, 1)):
        """
        Unnormalizes joint angles from a specified range back to the robot's joint limits.

        Args:
            config: Normalized joint configuration as numpy array or torch.Tensor. Can have dims
                  [DOF] if a single configuration
                  [B, DOF] if a batch of configurations
                  [B, T, DOF] if a batched time-series of configurations
            limits: Tuple of (min, max) values the config was normalized to, default (-1, 1)

        Returns:
            Unnormalized joint angles within the robot's joint limits, with same shape and type as input
        """
        # Handle torch tensors differently than numpy arrays
        if hasattr(config, "device") and hasattr(
            config, "dtype"
        ):  # It's a torch tensor

            assert isinstance(config, torch.Tensor), "Expected torch.Tensor"

            # Check input dimensions match the robot's DOF
            input_dof_dim = 0 if config.dim() == 1 else (1 if config.dim() == 2 else 2)
            input_dof = config.size(input_dof_dim)
            assert input_dof == self.DOF, f"Expected {self.DOF} DOF, got {input_dof}"

            assert torch.all(
                (config >= limits[0]) & (config <= limits[1])
            ), f"Normalized values must be in range [{limits[0]}, {limits[1]}]"

            # Get joint limits as a tensor
            joint_limits = torch.tensor(
                self.JOINT_LIMITS, dtype=config.dtype, device=config.device
            )

            # Calculate unnormalization parameters
            joint_range = joint_limits[:, 1] - joint_limits[:, 0]
            joint_min = joint_limits[:, 0]

            # Reshape joint limits if needed for broadcasting
            if config.dim() > 1:
                for _ in range(config.dim() - 1):
                    joint_range = joint_range.unsqueeze(0)
                    joint_min = joint_min.unsqueeze(0)

            # Unnormalize: first back to [0,1], then to joint limits
            unnormalized = (config - limits[0]) / (limits[1] - limits[0])
            unnormalized = unnormalized * joint_range + joint_min

            return unnormalized

        else:  # It's a numpy array or list
            # Ensure the config is a numpy array
            if not isinstance(config, np.ndarray):
                config = np.array(config)

            # Check input dimensions match the robot's DOF
            input_dof_dim = 0 if config.ndim == 1 else (1 if config.ndim == 2 else 2)
            input_dof = config.shape[input_dof_dim]
            assert input_dof == self.DOF, f"Expected {self.DOF} DOF, got {input_dof}"

            assert np.all(
                (config >= limits[0]) & (config <= limits[1])
            ), f"Normalized values must be in range [{limits[0]}, {limits[1]}]"

            # Get joint limits
            joint_limits = self.JOINT_LIMITS

            # Calculate unnormalization parameters
            joint_range = joint_limits[:, 1] - joint_limits[:, 0]
            joint_min = joint_limits[:, 0]

            # Reshape joint limits if needed for broadcasting
            if config.ndim > 1:
                for _ in range(config.ndim - 1):
                    joint_range = joint_range[np.newaxis, ...]
                    joint_min = joint_min[np.newaxis, ...]

            # Unnormalize: first back to [0,1], then to joint limits
            unnormalized = (config - limits[0]) / (limits[1] - limits[0])
            unnormalized = unnormalized * joint_range + joint_min

            return unnormalized

    def random_configuration(self) -> np.ndarray:
        """Generate a random configuration within joint limits."""
        limits = self.JOINT_LIMITS
        return (limits[:, 1] - limits[:, 0]) * np.random.rand(self.DOF) + limits[:, 0]

    def random_neutral(self, method: str = "normal") -> np.ndarray:
        """Generate a random configuration around the neutral pose."""
        if method == "normal":
            return np.clip(
                self.NEUTRAL + np.random.normal(0, 0.25, self.DOF),
                self.JOINT_LIMITS[:, 0],
                self.JOINT_LIMITS[:, 1],
            )
        if method == "uniform":
            # No need to clip this because it's always within range
            return self.NEUTRAL + np.random.uniform(0, 0.25, self.DOF)
        assert False, "method must be either normal or uniform"

    def get_joint_names(self):
        """Return the list of joint names."""
        return self.joint_names

    def get_link_names(self):
        """Return the list of link names."""
        return self.link_names

    def get_joint_limits(self):
        """Return the joint limits."""
        return self.JOINT_LIMITS

    def get_velocity_limits(self):
        """Return the velocity limits."""
        return self.VELOCITY_LIMIT

    def add_end_effector(self, name, transform=None):
        """
        Add a new end effector to the robot.

        Args:
            name: Name of the end effector link
            transform: SE3 transform from the default end effector to this one
        """
        if name not in self.link_names:
            raise ValueError(f"Link {name} not found in robot")

        self.EFF_LIST.add(name)
        if transform is not None:
            self.EFF_T_LIST[(self.default_eff, name)] = transform

    def fk(self, q: np.ndarray) -> SE3:
        """Forward kinematics for the robot.

        Args:
            q (np.ndarray): Joint configuration

        Returns:
            SE3: Transform from base to end effector
        """
        # TODO: Implement actual forward kinematics
        # For now, return a placeholder transform with floating-point arrays
        return SE3(
            pos=np.array([0.0, 0.0, 0.0], dtype=np.float64),
            quaternion=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        )

    def ik(self, pose, eff_frame=None):
        """
        This is a placeholder that just returns the neutral configuration

        TODO(William): Implement actual inverse kinematics or remove this
        method and related methods.

        Args:
            pose: SE3 pose of the end effector
            eff_frame: End effector frame to use

        Returns:
            List of joint configurations that achieve the pose
        """
        if eff_frame is None:
            eff_frame = self.default_eff

        assert eff_frame in self.EFF_LIST, f"End effector {eff_frame} not found"

        return [self.NEUTRAL]

    def random_ik(self, pose, eff_frame=None):
        """
        Generate a random IK solution for the given pose.

        Args:
            pose: SE3 pose of the end effector
            eff_frame: End effector frame to use

        Returns:
            A joint configuration that achieves the pose
        """
        if eff_frame is None:
            eff_frame = self.default_eff

        solutions = self.ik(pose, eff_frame)
        if not solutions:
            raise Exception(f"IK failed with {pose}")

        return solutions[0]

    def collision_free_ik(self, sim, sim_robot, pose, frame=None, retries=1000):
        """
        Find a collision-free IK solution.

        Args:
            sim: Simulation environment
            sim_robot: Robot in the simulation
            pose: SE3 pose of the end effector
            frame: End effector frame to use
            retries: Number of retries

        Returns:
            A collision-free joint configuration that achieves the pose
        """
        if frame is None:
            frame = self.default_eff

        for i in range(retries + 1):
            try:
                sample = self.random_ik(pose, frame)
                sim_robot.marionette(sample)
                if not sim.in_collision(sim_robot, check_self=True):
                    return sample
            except:
                continue
        return None


class FrankaRobot:
    # TODO(Adam) remove this after making this more general
    JOINT_LIMITS = np.array(
        [
            (-2.8973, 2.8973),
            (-1.7628, 1.7628),
            (-2.8973, 2.8973),
            (-3.0718, -0.0698),
            (-2.8973, 2.8973),
            (-0.0175, 3.7525),
            (-2.8973, 2.8973),
        ]
    )

    VELOCITY_LIMIT = np.array([2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100])
    ACCELERATION_LIMIT = np.array([15.0, 7.5, 10.0, 12.5, 15.0, 20.0, 20.0])
    DOF = 7
    EFF_LIST = set(["panda_link8", "right_gripper", "panda_grasptarget"])
    EFF_T_LIST = {
        ("panda_link8", "panda_hand"): SE3(
            pos=np.array([0, 0, 0]),
            quaternion=np.array([0.9238795325113726, 0.0, 0.0, -0.3826834323648827]),
        ),
        ("panda_link8", "right_gripper"): SE3(
            pos=np.array([0, 0, 0.1]), quaternion=SO3.from_rpy(0, 0, 2.35619449019).q
        ),
        ("panda_link8", "panda_grasptarget"): SE3(
            pos=np.array([0, 0, 0.105]),
            quaternion=SO3.from_rpy(0, 0, -0.785398163397).q,
        ),
        ("panda_hand", "right_gripper"): SE3(
            pos=np.array([0.0, 0.0, 0.1]), quaternion=np.array([0.0, 0.0, 0.0, 1.0])
        ),
    }
    # These are strings because that's needed for Bullet
    urdf = str(Path(__file__).parent / "urdf" / "franka_panda" / "panda.urdf")
    hd_urdf = str(Path(__file__).parent / "urdf" / "franka_panda" / "hd_panda.urdf")
    # This can be a Path because it's only ever used from Python
    pointcloud_cache = Path(__file__).parent / "pointcloud" / "cache" / "franka"
    NEUTRAL = np.array(
        [
            -0.017792060227770554,
            -0.7601235411041661,
            0.019782607023391807,
            -2.342050140544315,
            0.029840531355804868,
            1.5411935298621688,
            0.7534486589746342,
        ]
    )
    # Tuples of radius in meters and the corresponding links, values are centers on that link
    SPHERES = [
        (0.08, {"panda_link0": [[0.0, 0.0, 0.05]]}),
        (
            0.06,
            {
                "panda_link1": [
                    [0.0, -0.08, 0.0],
                    [0.0, -0.03, 0.0],
                    [0.0, 0.0, -0.12],
                    [0.0, 0.0, -0.17],
                ],
                "panda_link2": [
                    [0.0, 0.0, 0.03],
                    [0.0, 0.0, 0.08],
                    [0.0, -0.12, 0.0],
                    [0.0, -0.17, 0.0],
                ],
                "panda_link3": [[0.0, 0.0, -0.1]],
                "panda_link4": [[-0.08, 0.095, 0.0]],
                "panda_link5": [
                    [0.0, 0.055, 0.0],
                    [0.0, 0.075, 0.0],
                    [0.0, 0.0, -0.22],
                ],
            },
        ),
        (
            0.05,
            {
                "panda_link3": [[0.0, 0.0, -0.06]],
                "panda_link5": [[0.0, 0.05, -0.18]],
                "panda_link6": [[0.0, 0.0, 0.0], [0.08, -0.01, 0.0]],
                "panda_link7": [[0.0, 0.0, 0.07]],
            },
        ),
        (
            0.055,
            {
                "panda_link3": [[0.08, 0.06, 0.0], [0.08, 0.02, 0.0]],
                "panda_link4": [
                    [0.0, 0.0, 0.02],
                    [0.0, 0.0, 0.06],
                    [-0.08, 0.06, 0.0],
                ],
            },
        ),
        (
            0.025,
            {
                "panda_link5": [
                    [0.01, 0.08, -0.14],
                    [0.01, 0.085, -0.11],
                    [0.01, 0.09, -0.08],
                    [0.01, 0.095, -0.05],
                    [-0.01, 0.08, -0.14],
                    [-0.01, 0.085, -0.11],
                    [-0.01, 0.09, -0.08],
                    [-0.01, 0.095, -0.05],
                ],
                "panda_link7": [[0.02, 0.04, 0.08], [0.04, 0.02, 0.08]],
            },
        ),
        (0.052, {"panda_link6": [[0.08, 0.035, 0.0]]}),
        (0.02, {"panda_link7": [[0.04, 0.06, 0.085], [0.06, 0.04, 0.085]]}),
        (
            0.028,
            {
                "panda_hand": [
                    [0.0, -0.075, 0.01],
                    [0.0, -0.045, 0.01],
                    [0.0, -0.015, 0.01],
                    [0.0, 0.015, 0.01],
                    [0.0, 0.045, 0.01],
                    [0.0, 0.075, 0.01],
                ]
            },
        ),
        (
            0.026,
            {
                "panda_hand": [
                    [0.0, -0.075, 0.03],
                    [0.0, -0.045, 0.03],
                    [0.0, -0.015, 0.03],
                    [0.0, 0.015, 0.03],
                    [0.0, 0.045, 0.03],
                    [0.0, 0.075, 0.03],
                ]
            },
        ),
        (
            0.024,
            {
                "panda_hand": [
                    [0.0, -0.075, 0.05],
                    [0.0, -0.045, 0.05],
                    [0.0, -0.015, 0.05],
                    [0.0, 0.015, 0.05],
                    [0.0, 0.045, 0.05],
                    [0.0, 0.075, 0.05],
                ]
            },
        ),
        (
            0.012,
            {
                "panda_leftfinger": [
                    [0, 0.015, 0.022],
                    [0, 0.008, 0.044],
                ],
                "panda_rightfinger": [
                    [0, -0.015, 0.022],
                    [0, -0.008, 0.044],
                ],
            },
        ),
    ]

    @staticmethod
    def within_limits(config):
        # We have to add a small buffer because of float math
        return np.all(config >= FrankaRobot.JOINT_LIMITS[:, 0] - 1e-5) and np.all(
            config <= FrankaRobot.JOINT_LIMITS[:, 1] + 1e-5
        )

    @staticmethod
    def random_neutral(method="normal"):
        if method == "normal":
            return np.clip(
                FrankaRobot.NEUTRAL + np.random.normal(0, 0.25, 7),
                FrankaRobot.JOINT_LIMITS[:, 0],
                FrankaRobot.JOINT_LIMITS[:, 1],
            )
        if method == "uniform":
            # No need to clip this because it's always within range
            return FrankaRobot.NEUTRAL + np.random.uniform(0, 0.25, 7)
        assert False, "method must be either normal or uniform"

    @staticmethod
    def fk(config, eff_frame="right_gripper"):
        """
        Returns the SE3 frame of the end effector
        """
        assert (
            eff_frame in FrankaRobot.EFF_LIST
        ), "Default FK only calculated for a valid end effector frame"
        pos, rot = get_fk(config)
        mat = np.eye(4)
        mat[:3, :3] = np.asarray(rot)
        mat[:3, 3] = np.asarray(pos)
        if eff_frame == "panda_link8":
            return SE3(
                pos=np.asarray(pos), quaternion=SO3.from_matrix(np.asarray(rot)).q
            )
        elif eff_frame == "right_gripper":
            return (
                SE3(pos=np.asarray(pos), quaternion=SO3.from_matrix(np.asarray(rot)).q)
                @ FrankaRobot.EFF_T_LIST[("panda_link8", "right_gripper")]
            )
        else:
            return (
                SE3(pos=np.asarray(pos), quaternion=SO3.from_matrix(np.asarray(rot)).q)
                @ FrankaRobot.EFF_T_LIST[("panda_link8", "panda_grasptarget")]
            )

    @staticmethod
    def ik(pose, panda_link7, eff_frame="right_gripper"):
        """
        :param pose: SE3 pose expressed in specified end effector frame
        :param panda_link7: Value for the joint panda_link7, other IK can be calculated with this joint value set.
            Must be within joint range
        :param eff_frame: Desired end effector frame, must be among [panda_link8, right_gripper, panda_grasptarget]
        :return: Typically 4 solutions to IK
        """
        assert (
            eff_frame in FrankaRobot.EFF_LIST
        ), "IK only calculated for a valid end effector frame"
        if eff_frame == "right_gripper":
            pose = (
                pose @ FrankaRobot.EFF_T_LIST[("panda_link8", "right_gripper")].inverse
            )
        elif eff_frame == "panda_grasptarget":
            pose = (
                pose
                @ FrankaRobot.EFF_T_LIST[("panda_link8", "panda_grasptarget")].inverse
            )
        rot = pose.so3.matrix.tolist()
        pos = pose.xyz
        assert (
            panda_link7 >= FrankaRobot.JOINT_LIMITS[-1, 0]
            and panda_link7 <= FrankaRobot.JOINT_LIMITS[-1, 1]
        ), f"Value for floating joint must be within range {FrankaRobot.JOINT_LIMITS[-1, :].tolist()}"
        solutions = [np.asarray(s) for s in get_ik(pos, rot, [panda_link7])]
        return [
            s
            for s in solutions
            if (
                np.all(s >= FrankaRobot.JOINT_LIMITS[:, 0])
                and np.all(s <= FrankaRobot.JOINT_LIMITS[:, 1])
            )
        ]

    @staticmethod
    def random_configuration():
        limits = FrankaRobot.JOINT_LIMITS
        return (limits[:, 1] - limits[:, 0]) * (np.random.rand(7)) + limits[:, 0]

    @staticmethod
    def random_ik(pose, eff_frame="right_gripper"):
        config = FrankaRobot.random_configuration()
        try:
            return FrankaRobot.ik(pose, config[-1], eff_frame)
        except:
            raise Exception(f"IK failed with {pose}")

    @staticmethod
    def collision_free_ik(sim, sim_franka, pose, frame="right_gripper", retries=1000):
        for i in range(retries + 1):
            samples = FrankaRobot.random_ik(pose, "right_gripper")
            for sample in samples:
                sim_franka.marionette(sample)
                if not sim.in_collision(sim_franka, check_self=True):
                    return sample
        return None


class FrankaRealRobot(FrankaRobot):
    JOINT_LIMITS = np.array(
        [
            (-2.8973, 2.8973),
            (-1.7628, 1.7628),
            (-2.8973, 2.8973),
            (-3.0718, -0.0698),
            (-2.8973, 2.8973),
            (0.05, 3.75),
            (-2.8973, 2.8973),
        ]
    )

    @staticmethod
    def within_limits(config):
        # We have to add a small buffer because of float math
        return np.all(config >= FrankaRealRobot.JOINT_LIMITS[:, 0] - 1e-5) and np.all(
            config <= FrankaRealRobot.JOINT_LIMITS[:, 1] + 1e-5
        )

    @staticmethod
    def random_neutral(method="normal"):
        if method == "normal":
            return np.clip(
                FrankaRealRobot.NEUTRAL + np.random.normal(0, 0.25, 7),
                FrankaRealRobot.JOINT_LIMITS[:, 0],
                FrankaRealRobot.JOINT_LIMITS[:, 1],
            )
        if method == "uniform":
            # No need to clip this because it's always within range
            return FrankaRealRobot.NEUTRAL + np.random.uniform(0, 0.25, 7)
        assert False, "method must be either normal or uniform"

    @staticmethod
    def fk(config, eff_frame="right_gripper"):
        """
        Returns the SE3 frame of the end effector
        """
        assert (
            eff_frame in FrankaRealRobot.EFF_LIST
        ), "Default FK only calculated for a valid end effector frame"
        pos, rot = get_fk(config)
        mat = np.eye(4)
        mat[:3, :3] = np.asarray(rot)
        mat[:3, 3] = np.asarray(pos)
        if eff_frame == "panda_link8":
            return SE3(
                pos=np.asarray(pos), quaternion=SO3.from_matrix(np.asarray(rot)).q
            )
        elif eff_frame == "right_gripper":
            return (
                SE3(pos=np.asarray(pos), quaternion=SO3.from_matrix(np.asarray(rot)).q)
                @ FrankaRealRobot.EFF_T_LIST[("panda_link8", "right_gripper")]
            )
        else:
            return (
                SE3(pos=np.asarray(pos), quaternion=SO3.from_matrix(np.asarray(rot)).q)
                @ FrankaRealRobot.EFF_T_LIST[("panda_link8", "panda_grasptarget")]
            )

    @staticmethod
    def ik(pose, panda_link7, eff_frame="right_gripper"):
        """
        :param pose: SE3 pose expressed in specified end effector frame
        :param panda_link7: Value for the joint panda_link7, other IK can be calculated with this joint value set.
            Must be within joint range
        :param eff_frame: Desired end effector frame, must be among [panda_link8, right_gripper, panda_grasptarget]
        :return: Typically 4 solutions to IK
        """
        assert (
            eff_frame in FrankaRealRobot.EFF_LIST
        ), "IK only calculated for a valid end effector frame"
        if eff_frame == "right_gripper":
            pose = (
                pose
                @ FrankaRealRobot.EFF_T_LIST[("panda_link8", "right_gripper")].inverse
            )
        elif eff_frame == "panda_grasptarget":
            pose = (
                pose
                @ FrankaRealRobot.EFF_T_LIST[
                    ("panda_link8", "panda_grasptarget")
                ].inverse
            )
        rot = pose.so3.matrix.tolist()
        pos = pose.xyz
        assert (
            panda_link7 >= FrankaRealRobot.JOINT_LIMITS[-1, 0]
            and panda_link7 <= FrankaRealRobot.JOINT_LIMITS[-1, 1]
        ), f"Value for floating joint must be within range {FrankaRealRobot.JOINT_LIMITS[-1, :].tolist()}"
        solutions = [np.asarray(s) for s in get_ik(pos, rot, [panda_link7])]
        return [
            s
            for s in solutions
            if (
                np.all(s >= FrankaRealRobot.JOINT_LIMITS[:, 0])
                and np.all(s <= FrankaRealRobot.JOINT_LIMITS[:, 1])
            )
        ]

    @staticmethod
    def random_configuration():
        limits = FrankaRealRobot.JOINT_LIMITS
        return (limits[:, 1] - limits[:, 0]) * (np.random.rand(7)) + limits[:, 0]

    @staticmethod
    def random_ik(pose, eff_frame="right_gripper"):
        config = FrankaRealRobot.random_configuration()
        try:
            return FrankaRealRobot.ik(pose, config[-1], eff_frame)
        except:
            raise Exception(f"IK failed with {pose}")

    @staticmethod
    def collision_free_ik(
        sim, sim_franka, selfcc, pose, frame="right_gripper", retries=1000
    ):
        for i in range(retries + 1):
            samples = FrankaRealRobot.random_ik(pose, "right_gripper")
            for sample in samples:
                sim_franka.marionette(sample)
                if not (
                    sim.in_collision(sim_franka, check_self=True)
                    or selfcc.has_self_collision(sample)
                ):
                    return sample
        return None


class FrankaGripper:
    JOINT_LIMITS = None
    DOF = 6
    urdf = str(Path(__file__).parent / "urdf" / "panda_hand" / "panda.urdf")

    @staticmethod
    def random_configuration():
        raise NotImplementedError(
            "Random configuration not implemented for Franka Hand"
        )
