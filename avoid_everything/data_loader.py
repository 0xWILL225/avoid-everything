# MIT License
#
# Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES, University of Washington. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from pathlib import Path
from typing import Dict, Optional, Union

import lightning.pytorch as pl
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# from robofin.collision import FrankaCollisionSpheres
# from robofin.old.kinematics.numba import franka_arm_link_fk
# from old.robot_constants import RealFrankaConstants
# from old.samplers import NumpyFrankaSampler
from robofin.robots import Robot
from robofin.samplers import NumpyRobotSampler

from avoid_everything.dataset import Dataset as MPNDataset
from avoid_everything.geometry import construct_mixed_point_cloud
# from avoid_everything.normalization import normalize_franka_joints
from avoid_everything.type_defs import DatasetType

# Import for mpinets data format compatibility
try:
    from geometrout.primitive import Cuboid, Cylinder
except ImportError:
    Cuboid = None
    Cylinder = None


class Base(Dataset):
    """
    This base class should never be used directly, but it handles the filesystem
    management and the basic indexing. When using these dataloaders, the directory
    holding the data should look like so:
        directory/
          train/
             train.hdf5
          val/
             val.hdf5
          test/
             test.hdf5
    Note that only the relevant subdirectory is required, i.e. when creating a
    dataset for training, this class will not check for (and will not use) the val/
    and test/ subdirectories.
    """

    def __init__(
        self,
        robot: Robot,
        data_path: Union[Path, str],
        dataset_type: DatasetType,
        trajectory_key: str,
        num_robot_points: int,
        num_obstacle_points: int,
        num_target_points: int,
        random_scale: float,
    ):
        """
        :param robot (Robot): Robot object
        :param directory Path: The path to the root of the data directory
        :param num_robot_points int: The number of points to sample from the robot
        :param num_obstacle_points int: The number of points to sample from the obstacles
        :param num_target_points int: The number of points to sample from the target
                                      robot end effector
        :param dataset_type DatasetType: What type of dataset this is
        :param random_scale float: The standard deviation of the random normal
                                   noise to apply to the joints during training.
                                   This is only used for train datasets.
        """
        self.robot = robot
        self._database = Path(data_path)
        self.trajectory_key = trajectory_key
        self.train = dataset_type == DatasetType.TRAIN
        self.normalization_params = None
        if not self.file_exists:
            self.state_count = 0
            self.problem_count = 0
        else:
            # Try new format first, fall back to old mpinets format
            try:
                with MPNDataset(self._database) as f:
                    self.state_count = len(f[self.trajectory_key])
                    self.problem_count = len(f)
                self._use_mpinets_format = False
            except Exception:
                # Fall back to old mpinets format (direct h5py access)
                import h5py
                with h5py.File(str(self._database), "r") as f:
                    if self.trajectory_key in f:
                        traj_data = f[self.trajectory_key]
                        self.problem_count = traj_data.shape[0]
                        self.state_count = traj_data.shape[0] * traj_data.shape[1]  # num_problems * trajectory_length
                        self.expert_length = traj_data.shape[1]  # trajectory length
                    else:
                        self.state_count = 0
                        self.problem_count = 0
                        self.expert_length = 0
                self._use_mpinets_format = True

        self.num_obstacle_points = num_obstacle_points
        self.num_robot_points = num_robot_points
        self.num_target_points = num_target_points
        self.random_scale = random_scale
        self.robot_sampler = NumpyRobotSampler(
            self.robot,
            num_robot_points=self.num_robot_points,
            num_eef_points=self.num_target_points,
            use_cache=True,
            with_base_link=True,
        )

    @property
    def file_exists(self) -> bool:
        return self._database.exists()

    @property
    def md5_checksum(self):
        with MPNDataset(self._database) as f:
            return f.md5_checksum

    @classmethod
    def load_from_directory(
        cls,
        robot: Robot,
        directory: Union[Path, str],
        dataset_type: DatasetType,
        *args,
        **kwargs,
    ):
        directory = Path(directory)
        if dataset_type in (DatasetType.TRAIN, "train"):
            enclosing_path = directory / "train"
            data_path = enclosing_path / "train.hdf5"
        elif dataset_type in (DatasetType.VAL_STATE, "val"):
            enclosing_path = directory / "val"
            data_path = enclosing_path / "val.hdf5"
        elif dataset_type in (DatasetType.VAL, "val"):
            enclosing_path = directory / "val"
            data_path = enclosing_path / "val.hdf5"
        elif dataset_type in (DatasetType.MINI_TRAIN, "mini_train"):
            enclosing_path = directory / "val"
            data_path = enclosing_path / "mini_train.hdf5"
        elif dataset_type in (DatasetType.VAL_PRETRAIN, "val_pretrain"):
            enclosing_path = directory / "val"
            data_path = enclosing_path / "val_pretrain.hdf5"
        elif dataset_type in (DatasetType.TEST, "test"):
            enclosing_path = directory / "test"
            data_path = enclosing_path / "test.hdf5"
        else:
            raise Exception(f"Invalid dataset type: {dataset_type}")
        return cls(
            robot,
            data_path,
            dataset_type,
            *args,
            **kwargs,
        )

    def clamp_and_normalize(self, configuration_tensor: torch.Tensor):
        """
        Normalizes the joints between -1 and 1 according the the joint limits

        :param configuration_tensor (torch.Tensor): The input tensor. 
            Has dim [self.robot.MAIN_DOF]
        """
        # NOTE: self.robot.main_joint_limits based on URDF is different from the
        # original implementation's RealFrankaConstanst.JOINT_LIMITS for joint 6
        limits = torch.as_tensor(self.robot.main_joint_limits).float() 
        configuration_tensor = torch.minimum(
            torch.maximum(configuration_tensor, limits[:, 0]), limits[:, 1]
        )
        return self.robot.normalize_joints(configuration_tensor)

    def get_inputs_mpinets(self, trajectory_idx: int) -> Dict[str, torch.Tensor]:
        """
        Loads data from old mpinets format (direct h5py access)
        """
        item = {}
        import h5py
        with h5py.File(str(self._database), "r") as f:
            # Get target from last configuration in trajectory 
            target_config = f[self.trajectory_key][trajectory_idx, -1, :]
            assert isinstance(target_config, np.ndarray)
            target_pose = self.robot.fk(target_config)[self.robot.tcp_link_name]
            target_points = torch.as_tensor(
                self.robot_sampler.sample_end_effector(
                    target_pose,
                )[..., :3]
            ).float()
            item["target_position"] = torch.as_tensor(target_pose[:3, 3]).float()
            item["target_orientation"] = torch.as_tensor(target_pose[:3, :3]).float()

            # Load obstacle data - same structure as mpinets format
            item["cuboid_dims"] = torch.as_tensor(f["cuboid_dims"][trajectory_idx]).float()
            item["cuboid_centers"] = torch.as_tensor(f["cuboid_centers"][trajectory_idx]).float()
            cuboid_quats_tensor = torch.as_tensor(f["cuboid_quaternions"][trajectory_idx]).float()
            
            # Fix zero quaternions in tensor data before batching
            cuboid_quat_norms = torch.norm(cuboid_quats_tensor, dim=1)
            cuboid_invalid_mask = cuboid_quat_norms < 1e-8
            cuboid_quats_tensor[cuboid_invalid_mask] = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=cuboid_quats_tensor.dtype)
            item["cuboid_quats"] = cuboid_quats_tensor

            item["cylinder_radii"] = torch.as_tensor(f["cylinder_radii"][trajectory_idx]).float()
            item["cylinder_heights"] = torch.as_tensor(f["cylinder_heights"][trajectory_idx]).float()
            item["cylinder_centers"] = torch.as_tensor(f["cylinder_centers"][trajectory_idx]).float()
            cylinder_quats_tensor = torch.as_tensor(f["cylinder_quaternions"][trajectory_idx]).float()
            
            # Fix zero quaternions in tensor data before batching
            cylinder_quat_norms = torch.norm(cylinder_quats_tensor, dim=1)
            cylinder_invalid_mask = cylinder_quat_norms < 1e-8
            cylinder_quats_tensor[cylinder_invalid_mask] = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=cylinder_quats_tensor.dtype)
            item["cylinder_quats"] = cylinder_quats_tensor

            # Create obstacles from the data (similar to mpinets_data_loader.py)
            if Cuboid is None or Cylinder is None:
                raise ImportError("geometrout.primitive is required for mpinets format compatibility")
            
            cuboid_dims = f["cuboid_dims"][trajectory_idx]
            cuboid_centers = f["cuboid_centers"][trajectory_idx]
            cuboid_quats = f["cuboid_quaternions"][trajectory_idx]
            
            # Handle dimension expansion for single obstacles
            if cuboid_dims.ndim == 1:
                cuboid_dims = np.expand_dims(cuboid_dims, axis=0)
            if cuboid_centers.ndim == 1:
                cuboid_centers = np.expand_dims(cuboid_centers, axis=0)
            if cuboid_quats.ndim == 1:
                cuboid_quats = np.expand_dims(cuboid_quats, axis=0)
            
            # Fix invalid quaternions (from mpinets format)
            # Entries without a shape are stored with an invalid quaternion of all zeros
            # This will cause NaNs later in the pipeline. It's best to set these to unit
            # quaternions.
            # To find invalid shapes, we just look for a dimension with size 0
            # Also fix quaternions with very small norms that would cause NaN during normalization
            zero_quat_mask = np.all(np.isclose(cuboid_quats, 0), axis=1)
            small_norm_mask = np.linalg.norm(cuboid_quats, axis=1) < 1e-8
            invalid_quat_mask = np.logical_or(zero_quat_mask, small_norm_mask)
            cuboid_quats[invalid_quat_mask, 0] = 1
            cuboid_quats[invalid_quat_mask, 1:] = 0
            
            # Filter out zero-volume cuboids
            cuboids = []
            for i in range(len(cuboid_dims)):
                if np.any(cuboid_dims[i] > 0):  # Non-zero volume
                    cuboids.append(Cuboid(cuboid_centers[i], cuboid_dims[i], cuboid_quats[i]))
            
            cylinder_radii = f["cylinder_radii"][trajectory_idx]
            cylinder_heights = f["cylinder_heights"][trajectory_idx]
            cylinder_centers = f["cylinder_centers"][trajectory_idx]
            cylinder_quats = f["cylinder_quaternions"][trajectory_idx]
            
            # Handle dimension expansion for single obstacles
            if cylinder_radii.ndim == 1:
                cylinder_radii = np.expand_dims(cylinder_radii, axis=0)
            if cylinder_heights.ndim == 1:
                cylinder_heights = np.expand_dims(cylinder_heights, axis=0)
            if cylinder_centers.ndim == 1:
                cylinder_centers = np.expand_dims(cylinder_centers, axis=0)
            if cylinder_quats.ndim == 1:
                cylinder_quats = np.expand_dims(cylinder_quats, axis=0)
            
            # Fix invalid quaternions (from mpinets format)
            # Ditto to the comment above about fixing ill-formed quaternions
            # Also fix quaternions with very small norms that would cause NaN during normalization
            zero_quat_mask = np.all(np.isclose(cylinder_quats, 0), axis=1)
            small_norm_mask = np.linalg.norm(cylinder_quats, axis=1) < 1e-8
            invalid_quat_mask = np.logical_or(zero_quat_mask, small_norm_mask)
            cylinder_quats[invalid_quat_mask, 0] = 1
            cylinder_quats[invalid_quat_mask, 1:] = 0
            
            # Filter out zero-volume cylinders
            cylinders = []
            for i in range(len(cylinder_radii)):
                if cylinder_radii[i] > 0 and cylinder_heights[i] > 0:
                    cylinders.append(Cylinder(cylinder_centers[i], cylinder_radii[i].item(), cylinder_heights[i].item(), cylinder_quats[i]))
            
            obstacles = cuboids + cylinders
            scene_points = torch.as_tensor(
                construct_mixed_point_cloud(obstacles, self.num_obstacle_points)[..., :3]
            ).float()
            
            item["point_cloud"] = torch.cat((scene_points, target_points), dim=0)
            item["point_cloud_labels"] = torch.cat(
                (
                    torch.ones(len(scene_points), 1),
                    2 * torch.ones(len(target_points), 1),
                )
            )

        return item

    def get_inputs(self, problem, flobs) -> Dict[str, torch.Tensor]:
        """
        Loads all the relevant data and puts it in a dictionary. This includes
        normalizing all configurations and constructing the pointcloud.
        If a training dataset, applies some randomness to joints (before
        sampling the pointcloud).

        :param trajectory_idx int: The index of the trajectory in the hdf5 file
        :param timestep int: The timestep within that trajectory
        :rtype Dict[str, torch.Tensor]: The data used aggregated by the dataloader
                                        and used for training
        """
        item = {}
        target_pose = self.robot.fk(problem.target)[self.robot.tcp_link_name]
        target_points = torch.as_tensor(
            self.robot_sampler.sample_end_effector(
                target_pose,
            )[..., :3]
        ).float()
        item["target_position"] = torch.as_tensor(target_pose[:3, 3]).float()
        item["target_orientation"] = torch.as_tensor(target_pose[:3, :3]).float()

        item["cuboid_dims"] = torch.as_tensor(flobs.cuboid_dims).float()
        item["cuboid_centers"] = torch.as_tensor(flobs.cuboid_centers).float()
        item["cuboid_quats"] = torch.as_tensor(flobs.cuboid_quaternions).float()

        item["cylinder_radii"] = torch.as_tensor(flobs.cylinder_radii).float()
        item["cylinder_heights"] = torch.as_tensor(flobs.cylinder_heights).float()
        item["cylinder_centers"] = torch.as_tensor(flobs.cylinder_centers).float()
        item["cylinder_quats"] = torch.as_tensor(flobs.cylinder_quaternions).float()

        scene_points = torch.as_tensor(
            construct_mixed_point_cloud(problem.obstacles, self.num_obstacle_points)[
                ..., :3
            ]
        ).float()
        item["point_cloud"] = torch.cat((scene_points, target_points), dim=0)
        item["point_cloud_labels"] = torch.cat(
            (
                torch.ones(len(scene_points), 1),
                2 * torch.ones(len(target_points), 1),
            )
        )

        return item


class TrajectoryDataset(Base):
    """
    This dataset is used exclusively for validating. Each element in the dataset
    represents a trajectory start and scene. There is no supervision because
    this is used to produce an entire rollout and check for success. When doing
    validation, we care more about success than we care about matching the
    expert's behavior (which is a key difference from training).
    """

    def __init__(
        self,
        robot: Robot,
        data_path: Union[Path, str],
        dataset_type: DatasetType,
        trajectory_key: str,
        num_robot_points: int,
        num_obstacle_points: int,
        num_target_points: int,
        random_scale: float = 0.0,
    ):
        """
        :param directory Path: The path to the root of the data directory
        :param num_robot_points int: The number of points to sample from the robot
        :param num_obstacle_points int: The number of points to sample from the obstacles
        :param num_target_points int: The number of points to sample from the target
                                      robot end effector
        :param dataset_type DatasetType: What type of dataset this is
        """
        super().__init__(
            robot,
            data_path,
            dataset_type,
            trajectory_key,
            num_robot_points,
            num_obstacle_points,
            num_target_points,
            random_scale,
        )

    @classmethod
    def load_from_directory(
        cls,
        robot: Robot,
        directory: Path,
        trajectory_key: str,
        num_robot_points: int,
        num_obstacle_points: int,
        num_target_points: int,
        dataset_type: DatasetType,
        random_scale: float,
    ):
        return super().load_from_directory(
            robot,
            directory,
            dataset_type,
            trajectory_key,
            num_robot_points,
            num_obstacle_points,
            num_target_points,
            random_scale,
        )

    def __len__(self):
        """
        Necessary for Pytorch. For this dataset, the length is the total number
        of problems
        """
        return self.problem_count

    def unpadded_expert(self, pidx: int):
        with MPNDataset(self._database, "r") as f:
            return torch.as_tensor(f[self.trajectory_key].expert(pidx))

    def __getitem__(self, pidx: int) -> Dict[str, torch.Tensor]:
        """
        Required by Pytorch. Queries for data at a particular index. Note that
        in this dataset, the index always corresponds to the trajectory index.

        :param pidx int: The problem index
        :rtype Dict[str, torch.Tensor]: Returns a dictionary that can be assembled
            by the data loader before using in training.
        """
        if self._use_mpinets_format:
            # Handle old mpinets format
            import h5py
            item = self.get_inputs_mpinets(pidx)
            
            with h5py.File(str(self._database), "r") as f:
                # Get initial configuration (first timestep)
                config = f[self.trajectory_key][pidx, 0, :]
                config_tensor = torch.as_tensor(config).float()

                if self.train:
                    # Add slight random noise to the joints
                    randomized = (
                        self.random_scale * torch.randn(config_tensor.shape) + config_tensor
                    )

                    item["configuration"] = self.clamp_and_normalize(randomized)
                    robot_points = self.robot_sampler.sample(
                        randomized.numpy()
                    )[:, :3]
                else:
                    item["configuration"] = self.clamp_and_normalize(config_tensor)
                    robot_points = self.robot_sampler.sample(
                        config_tensor.numpy(),
                    )[:, :3]
                robot_points = torch.as_tensor(robot_points).float()

                item["point_cloud"] = torch.cat((robot_points, item["point_cloud"]), dim=0)
                item["point_cloud_labels"] = torch.cat(
                    (
                        torch.zeros(len(robot_points), 1),
                        item["point_cloud_labels"],
                    )
                )
                # Get full expert trajectory
                item["expert"] = torch.as_tensor(f[self.trajectory_key][pidx])
            item["pidx"] = torch.as_tensor(pidx)
            return item
        else:
            # Handle new format
            with MPNDataset(self._database, "r") as f:
                problem = f[self.trajectory_key].problem(pidx)
                flobs = f[self.trajectory_key].flattened_obstacles(pidx)
                item = self.get_inputs(problem, flobs)
                config = f[self.trajectory_key].problem(pidx).q0
                config_tensor = torch.as_tensor(config).float()

                if self.train:
                    # Add slight random noise to the joints
                    randomized = (
                        self.random_scale * torch.randn(config_tensor.shape) + config_tensor
                    )

                    item["configuration"] = self.clamp_and_normalize(randomized)
                    robot_points = self.robot_sampler.sample(
                        randomized.numpy()
                    )[:, :3]
                else:
                    item["configuration"] = self.clamp_and_normalize(config_tensor)
                    robot_points = self.robot_sampler.sample(
                        config_tensor.numpy(),
                    )[:, :3]
                robot_points = torch.as_tensor(robot_points).float()

                item["point_cloud"] = torch.cat((robot_points, item["point_cloud"]), dim=0)
                item["point_cloud_labels"] = torch.cat(
                    (
                        torch.zeros(len(robot_points), 1),
                        item["point_cloud_labels"],
                    )
                )
                item["expert"] = torch.as_tensor(f[self.trajectory_key].padded_expert(pidx))
            item["pidx"] = torch.as_tensor(pidx)

            return item


class StateDataset(Base):
    """
    This is the dataset used primarily for training. Each element in the dataset
    represents the robot and scene at a particular time $t$. Likewise, the
    supervision is the robot's configuration at q_{t+1}.
    """

    def __init__(
        self,
        robot: Robot,
        data_path: Union[Path, str],
        dataset_type: DatasetType,
        trajectory_key: str,
        num_robot_points: int,
        num_obstacle_points: int,
        num_target_points: int,
        random_scale: float,
        action_chunk_length: int,
    ):
        """
        :param directory Path: The path to the root of the data directory
        :param num_robot_points int: The number of points to sample from the robot
        :param num_obstacle_points int: The number of points to sample from the obstacles
        :param num_target_points int: The number of points to sample from the target
                                      robot end effector
        :param dataset_type DatasetType: What type of dataset this is
        :param random_scale float: The standard deviation of the random normal
                                   noise to apply to the joints during training.
                                   This is only used for train datasets.
        """
        super().__init__(
            robot,
            data_path,
            dataset_type,
            trajectory_key,
            num_robot_points,
            num_obstacle_points,
            num_target_points,
            random_scale,
        )
        self.action_chunk_length = action_chunk_length

    @classmethod
    def load_from_directory(
        cls,
        robot: Robot,
        directory: Path,
        trajectory_key: str,
        num_robot_points: int,
        num_obstacle_points: int,
        num_target_points: int,
        dataset_type: DatasetType,
        random_scale: float,
        action_chunk_length: int,
    ):
        return super().load_from_directory(
            robot,
            directory,
            dataset_type,
            trajectory_key,
            num_robot_points,
            num_obstacle_points,
            num_target_points,
            random_scale,
            action_chunk_length=action_chunk_length,
        )

    def __len__(self):
        """
        Returns the total number of start configurations in the dataset (i.e.
        the length of the trajectories times the number of trajectories)

        """
        return self.state_count

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns a training datapoint representing a single configuration in a
        single scene with the configuration at the next timestep as supervision

        :param idx int: Index represents the timestep within the trajectory
        :rtype Dict[str, torch.Tensor]: The data used for training
        """
        if self._use_mpinets_format:
            # Handle old mpinets format
            import h5py
            with h5py.File(str(self._database), "r") as f:
                # Calculate trajectory and timestep from flat index
                trajectory_idx, timestep = divmod(idx, self.expert_length)
                
                item = self.get_inputs_mpinets(trajectory_idx)
                
                # Get current configuration
                config = f[self.trajectory_key][trajectory_idx, timestep, :]
                config_tensor = torch.as_tensor(config).float()

                # Get supervision (next timesteps)
                supervision_timesteps = []
                for i in range(1, self.action_chunk_length + 1):
                    sup_timestep = min(timestep + i, self.expert_length - 1)
                    supervision_timesteps.append(f[self.trajectory_key][trajectory_idx, sup_timestep, :])
                supervision = np.array(supervision_timesteps)

                if self.train:
                    # Add slight random noise to the joints
                    randomized = (
                        self.random_scale * torch.randn(config_tensor.shape) + config_tensor
                    )

                    item["configuration"] = self.clamp_and_normalize(randomized)
                    robot_points = self.robot_sampler.sample(
                        randomized.numpy()
                    )[:, :3]
                else:
                    item["configuration"] = self.clamp_and_normalize(config_tensor)
                    robot_points = self.robot_sampler.sample(
                        config_tensor.numpy()
                    )[:, :3]
                robot_points = torch.as_tensor(robot_points).float()
                item["point_cloud"] = torch.cat((robot_points, item["point_cloud"]), dim=0)
                item["point_cloud_labels"] = torch.cat(
                    (
                        torch.zeros(len(robot_points), 1),
                        item["point_cloud_labels"],
                    )
                )

                item["idx"] = torch.as_tensor(idx)
                supervision_tensor = torch.as_tensor(supervision).float()
                item["supervision"] = self.clamp_and_normalize(supervision_tensor)

            return item
        else:
            # Handle new format
            with MPNDataset(self._database, "r") as f:
                pidx = f[self.trajectory_key].lookup_pidx(idx)
                problem = f[self.trajectory_key].problem(pidx)
                flobs = f[self.trajectory_key].flattened_obstacles(pidx)
                item = self.get_inputs(problem, flobs)
                configs = f[self.trajectory_key].state_range(
                    idx, lookahead=self.action_chunk_length + 1
                )
                config = configs[0]
                supervision = configs[1:]
                config_tensor = torch.as_tensor(config).float()

                if self.train:
                    # Add slight random noise to the joints
                    randomized = (
                        self.random_scale * torch.randn(config_tensor.shape) + config_tensor
                    )

                    item["configuration"] = self.clamp_and_normalize(randomized)
                    robot_points = self.robot_sampler.sample(
                        randomized.numpy()
                    )[:, :3]
                else:
                    item["configuration"] = self.clamp_and_normalize(config_tensor)
                    robot_points = self.robot_sampler.sample(
                        config_tensor.numpy()
                    )[:, :3]
                robot_points = torch.as_tensor(robot_points).float()
                item["point_cloud"] = torch.cat((robot_points, item["point_cloud"]), dim=0)
                item["point_cloud_labels"] = torch.cat(
                    (
                        torch.zeros(len(robot_points), 1),
                        item["point_cloud_labels"],
                    )
                )

                item["idx"] = torch.as_tensor(idx)
                supervision_tensor = torch.as_tensor(supervision).float()
                item["supervision"] = self.clamp_and_normalize(supervision_tensor)

            return item


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        robot: Robot,
        data_dir: str,
        train_trajectory_key: str,
        val_trajectory_key: str,
        num_robot_points: int,
        num_obstacle_points: int,
        num_target_points: int,
        action_chunk_length: int,
        random_scale: float,
        train_batch_size: int,
        val_batch_size: int,
        num_workers: int,
    ):
        super().__init__()
        self.robot = robot
        self.data_dir = Path(data_dir)
        self.train_trajectory_key = train_trajectory_key
        self.val_trajectory_key = val_trajectory_key
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.num_robot_points = num_robot_points
        self.num_obstacle_points = num_obstacle_points
        self.num_target_points = num_target_points
        self.num_workers = num_workers
        self.random_scale = random_scale
        self.action_chunk_length = action_chunk_length

    def setup(self, stage: Optional[str] = None):
        """
        A Pytorch Lightning method that is called per-device in when doing
        distributed training.

        :param stage Optional[str]: Indicates whether we are in the training
                                    procedure or if we are doing ad-hoc testing
        """
        if stage == "fit" or stage is None:
            self.data_train = StateDataset.load_from_directory(
                self.robot,
                self.data_dir,
                dataset_type=DatasetType.TRAIN,
                trajectory_key=self.train_trajectory_key,
                num_robot_points=self.num_robot_points,
                num_obstacle_points=self.num_obstacle_points,
                num_target_points=self.num_target_points,
                random_scale=self.random_scale,
                action_chunk_length=self.action_chunk_length,
            )
            self.data_val_state = StateDataset.load_from_directory(
                self.robot,
                self.data_dir,
                dataset_type=DatasetType.VAL_STATE,
                trajectory_key=self.val_trajectory_key,
                num_robot_points=self.num_robot_points,
                num_obstacle_points=self.num_obstacle_points,
                num_target_points=self.num_target_points,
                random_scale=0.0,
                action_chunk_length=self.action_chunk_length,
            )
            self.data_val = TrajectoryDataset.load_from_directory(
                self.robot,
                self.data_dir,
                dataset_type=DatasetType.VAL,
                trajectory_key=self.val_trajectory_key,
                num_robot_points=self.num_robot_points,
                num_obstacle_points=self.num_obstacle_points,
                num_target_points=self.num_target_points,
                random_scale=0.0,
            )
            # Handle missing optional validation files gracefully
            mini_train_path = Path(self.data_dir) / "val" / "mini_train.hdf5"
            if mini_train_path.exists():
                try:
                    self.data_mini_train = TrajectoryDataset.load_from_directory(
                        self.robot,
                        self.data_dir,
                        dataset_type=DatasetType.MINI_TRAIN,
                        trajectory_key=self.val_trajectory_key,
                        num_robot_points=self.num_robot_points,
                        num_obstacle_points=self.num_obstacle_points,
                        num_target_points=self.num_target_points,
                        random_scale=0.0,
                    )
                except:
                    self.data_mini_train = self.data_val
            else:
                # Use validation data as fallback for mini_train if file doesn't exist
                self.data_mini_train = self.data_val
                
            val_pretrain_path = Path(self.data_dir) / "val" / "val_pretrain.hdf5"
            if val_pretrain_path.exists():
                try:
                    self.data_val_pretrain = TrajectoryDataset.load_from_directory(
                        self.robot,
                        self.data_dir,
                        dataset_type=DatasetType.VAL_PRETRAIN,
                        trajectory_key=self.val_trajectory_key,
                        num_robot_points=self.num_robot_points,
                        num_obstacle_points=self.num_obstacle_points,
                        num_target_points=self.num_target_points,
                        random_scale=0.0,
                    )
                except:
                    self.data_val_pretrain = self.data_val
            else:
                # Use validation data as fallback for val_pretrain if file doesn't exist
                self.data_val_pretrain = self.data_val
        if stage == "test" or stage is None:
            self.data_test = StateDataset.load_from_directory(
                self.robot,
                self.data_dir,
                self.train_trajectory_key,  # TODO change this
                self.num_robot_points,
                self.num_obstacle_points,
                self.num_target_points,
                dataset_type=DatasetType.TEST,
                random_scale=self.random_scale,
                action_chunk_length=self.action_chunk_length,
            )
        if stage == "dagger":
            self.data_dagger = TrajectoryDataset.load_from_directory(
                self.robot,
                self.data_dir,
                dataset_type=DatasetType.TRAIN,
                trajectory_key=self.val_trajectory_key,
                num_robot_points=self.num_robot_points,
                num_obstacle_points=self.num_obstacle_points,
                num_target_points=self.num_target_points,
                random_scale=0.0,
            )

    def train_dataloader(self) -> DataLoader:
        """
        A Pytorch lightning method to get the dataloader for training

        :rtype DataLoader: The training dataloader
        """
        return DataLoader(
            self.data_train,
            self.train_batch_size,
            num_workers=self.num_workers,
            pin_memory=True,  # Restored significantly
            shuffle=True,
        )

    def dagger_dataloader(self) -> DataLoader:
        return DataLoader(
            self.data_dagger,
            self.val_batch_size,
            num_workers=self.num_workers,
            pin_memory=True,  # Restored
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        """
        A Pytorch lightning method to get the dataloader for validation

        :rtype DataLoader: The validation dataloader
        """
        loaders = [None, None, None, None]
        loaders[DatasetType.VAL_STATE] = DataLoader(
            self.data_val_state,
            self.train_batch_size,
            num_workers=self.num_workers,
            pin_memory=True, 
        )
        loaders[DatasetType.VAL] = DataLoader(
            self.data_val,
            self.val_batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        loaders[DatasetType.MINI_TRAIN] = DataLoader(
            self.data_mini_train,
            self.val_batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        loaders[DatasetType.VAL_PRETRAIN] = DataLoader(
            self.data_val_pretrain,
            self.val_batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        return loaders

    def test_dataloader(self) -> DataLoader:
        """
        A Pytorch lightning method to get the dataloader for testing

        :rtype DataLoader: The dataloader for testing
        """
        assert NotImplementedError("Not implemented")

    def md5_checksums(self):
        """
        Currently too lazy to figure out how to fit this into Lightning with the whole
        setup() thing and the data being initialized in that call and when to get
        hyperparameters etc etc, so just hardcoding the paths right now
        """
        paths = [
            ("train", self.data_dir / "train" / "train.hdf5"),
            ("val", self.data_dir / "val" / "val.hdf5"),
            ("mini_train", self.data_dir / "val" / "mini_train.hdf5"),
        ]
        checksums = {}
        for key, path in paths:
            if path.exists():
                with MPNDataset(path) as f:
                    checksums[key] = f.md5_checksum
        return checksums
