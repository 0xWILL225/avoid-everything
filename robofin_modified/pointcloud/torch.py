from pathlib import Path
import logging

import torch
import numpy as np
import trimesh

from robofin.torch_urdf import TorchURDF
from robofin.robots import Robot
from robofin.collision import FrankaSelfCollisionSampler as NumpySelfCollisionSampler
from geometrout.primitive import Sphere


def transform_pointcloud(pc, transformation_matrix, in_place=True):
    """

    Parameters
    ----------
    pc: A pytorch tensor pointcloud, maybe with some addition dimensions.
        This should have shape N x [3 + M] where N is the number of points
        M could be some additional mask dimensions or whatever, but the
        3 are x-y-z
    transformation_matrix: A 4x4 homography

    Returns
    -------
    Mutates the pointcloud in place and transforms x, y, z according the homography

    """
    assert isinstance(pc, torch.Tensor)
    assert type(pc) == type(transformation_matrix)
    assert pc.ndim == transformation_matrix.ndim
    if pc.ndim == 3:
        N, M = 1, 2
    elif pc.ndim == 2:
        N, M = 0, 1
    else:
        raise Exception("Pointcloud must have dimension Nx3 or BxNx3")

    xyz = pc[..., :3]

    ones_dim = list(xyz.shape)
    ones_dim[-1] = 1
    ones_dim = tuple(ones_dim)
    homogeneous_xyz = torch.cat((xyz, torch.ones(ones_dim, device=xyz.device)), dim=M)

    transformed_xyz = torch.matmul(
        transformation_matrix, homogeneous_xyz.transpose(N, M)
    )

    if in_place:
        pc[..., :3] = transformed_xyz[..., :3, :].transpose(N, M)
        return pc
    return torch.cat((transformed_xyz[..., :3, :].transpose(N, M), pc[..., 3:]), dim=M)


class RobotSampler:
    """
    This class allows for fast pointcloud sampling from the surface of a robot.
    At initialization, it loads a URDF and samples points from the mesh of each link.
    The points per link are based on the (very approximate) surface area of the link.

    Then, after instantiation, the sample method takes in a batch of configurations
    and produces pointclouds for each configuration by running FK on a subsample
    of the per-link pointclouds that are established at initialization.
    """

    def __init__(
        self,
        device,
        robot: Robot,
        num_fixed_points=None,
        use_cache=False,
        default_prismatic_value=0.025,
        with_base_link=True,
        deterministic_sampling=False,
    ):
        """
        Initialize the robot sampler.

        Parameters
        ----------
        device : torch.device
            Device to place tensors on.
        robot : Robot
            Robot object.
        num_fixed_points : int, optional
            Number of fixed points to sample. If None, uses all available points.
        use_cache : bool, optional
            Whether to use cached point clouds.
        default_prismatic_value : float, optional
            Default value for prismatic joints.
        with_base_link : bool, optional
            Whether to include the base link in sampling.
        deterministic_sampling : bool, optional
            Whether to use deterministic sampling for consistent results.
        """
        logging.getLogger("trimesh").setLevel("ERROR")
        self.device = device
        self.robot = robot
        self.num_fixed_points = num_fixed_points
        self.use_cache = use_cache
        self.default_prismatic_value = default_prismatic_value
        self.with_base_link = with_base_link
        self.deterministic_sampling = deterministic_sampling
        # Store a fixed seed for deterministic sampling
        self._rng_seed = np.random.randint(0, 2**32)
        self.rng = np.random.RandomState(self._rng_seed)
        self._init_internal_(device, use_cache)

    def _init_internal_(self, device, use_cache):
        """Initialize the internal state of the sampler."""
        self.torch_robot = TorchURDF.load(
            self.robot.urdf_path, lazy_load_meshes=True, device=device
        )
        self.links = [l for l in self.torch_robot.links if len(l.visuals)]

        # Create a cache directory for this robot if it doesn't exist
        self.pointcloud_cache = (
            Path(self.robot.urdf_path).parent.parent
            / "pointcloud"
            / "cache"
            / self.robot.name
        )
        self.pointcloud_cache.mkdir(parents=True, exist_ok=True)

        if use_cache and self._init_from_cache_(device):
            return

        # Load meshes for each link
        meshes = []
        for l in self.links:
            if hasattr(l.visuals[0].geometry, "mesh") and hasattr(
                l.visuals[0].geometry.mesh, "filename"
            ):
                mesh_filename = l.visuals[0].geometry.mesh.filename
                # Remove the "file:" prefix if it exists
                if mesh_filename.startswith("file:"):
                    mesh_filename = mesh_filename[5:]
                # If the path is absolute, use it directly
                if mesh_filename.startswith("/"):
                    mesh_path = Path(mesh_filename)
                else:
                    mesh_path = Path(self.robot.urdf_path).parent / mesh_filename
                if mesh_path.exists():
                    meshes.append(trimesh.load(str(mesh_path), force="mesh"))
                else:
                    print(f"Warning: Mesh file not found: {mesh_path}")
                    # If mesh file doesn't exist, create a simple box as placeholder
                    meshes.append(trimesh.creation.box(extents=[0.1, 0.1, 0.1]))
            else:
                # For links without mesh, create a simple box as placeholder
                meshes.append(trimesh.creation.box(extents=[0.1, 0.1, 0.1]))

        # Calculate areas and number of points per link
        areas = [mesh.bounding_box_oriented.area for mesh in meshes]
        if self.num_fixed_points is not None:
            num_points = np.round(
                self.num_fixed_points * np.array(areas) / np.sum(areas)
            )
            num_points[0] += self.num_fixed_points - np.sum(
                num_points
            )  # adjust for points lost to rounding
            assert np.sum(num_points) == self.num_fixed_points
        else:
            num_points = np.round(4096 * np.array(areas) / np.sum(areas))

        # Sample points from each mesh
        self.points = {}
        for ii in range(len(meshes)):
            pc = trimesh.sample.sample_surface(meshes[ii], int(num_points[ii]))[0]
            self.points[self.links[ii].name] = torch.as_tensor(
                pc, device=device
            ).unsqueeze(0)

        # If we made it all the way here with the use_cache flag set,
        # then we should be creating new cache files locally
        if use_cache:
            points_to_save = {
                k: tensor.squeeze(0).cpu().numpy() for k, tensor in self.points.items()
            }
            file_name = self._get_cache_file_name_()
            print(
                "\033[36m"
                + f"Saving new file to cache: {file_name}"
                + " from device: "
                + str(device)
                + "\033[0m"
            )
            np.save(file_name, points_to_save)

    def _get_cache_file_name_(self):
        """Get the cache file name for the current robot."""
        if self.num_fixed_points is not None:
            return (
                self.pointcloud_cache / f"fixed_point_cloud_{self.num_fixed_points}.npy"
            )
        else:
            return self.pointcloud_cache / "full_point_cloud.npy"

    def _init_from_cache_(self, device):
        """Initialize from cache if available."""
        file_name = self._get_cache_file_name_()
        if not file_name.is_file():
            return False

        points = np.load(
            file_name,
            allow_pickle=True,
        )
        self.points = {
            key: torch.as_tensor(pc, device=device).unsqueeze(0)
            for key, pc in points.item().items()
        }

        print(
            "\033[32m"
            + "Loaded points from cache: "
            + str(file_name)
            + " on device: "
            + str(device)
            + "\033[0m"
        )

        return True

    def end_effector_pose(self, config, frame=None, normalized=False):
        """Get the end effector pose for a given configuration."""
        if frame is None:
            frame = self.robot.default_eff

        if config.ndim == 1:
            config = config.unsqueeze(0)

        if normalized:
            config = self.robot.unnormalize_joints(config)

        # Handle prismatic joints if needed
        if hasattr(self.robot, "prismatic_joints") and self.robot.prismatic_joints:
            # Add default values for prismatic joints
            cfg = torch.cat(
                (
                    config,
                    self.default_prismatic_value
                    * torch.ones(
                        (config.shape[0], len(self.robot.prismatic_joints)),
                        device=config.device,
                    ),
                ),
                dim=1,
            )
        else:
            cfg = config

        fk = self.torch_robot.link_fk_batch(cfg, use_names=True)
        return fk[frame]

    def sample_end_effector(self, poses, num_points=None, frame=None):
        """
        Sample points from the end effector.

        Args:
            poses: The poses to sample from
            num_points: Number of points to sample. If None, uses self.num_fixed_points.
            frame: The end effector frame to use

        Returns:
            A tensor of sampled points
        """
        if frame is None:
            frame = self.robot.default_eff

        assert poses.ndim in [2, 3]
        if poses.ndim == 2:
            poses = poses.unsqueeze(0)

        # Get the default configuration
        default_cfg = torch.zeros((1, self.robot.DOF), device=poses.device)

        # Get the forward kinematics
        fk = self.torch_robot.visual_geometry_fk_batch(default_cfg)

        # Find end effector link
        eff_link_names = [frame]  # Just use the end effector frame

        # Get the transform for the end effector
        values = [
            list(fk.values())[idx]
            for idx, l in enumerate(self.links)
            if l.name in eff_link_names
        ]
        end_effector_links = [l for l in self.links if l.name in eff_link_names]

        # If we found the end effector link
        if end_effector_links and frame in self.points:
            # Apply transform to the end effector points
            pc = transform_pointcloud(
                self.points[frame].type_as(poses).repeat(poses.size(0), 1, 1),
                poses,
                in_place=True,
            )
        else:
            # If no points were found, create a simple point cloud at the end effector
            default_points = torch.tensor(
                [
                    [0.0, 0.0, 0.0],  # Origin
                    [0.1, 0.0, 0.0],  # X axis
                    [0.0, 0.1, 0.0],  # Y axis
                    [0.0, 0.0, 0.1],  # Z axis
                ],
                device=poses.device,
            ).unsqueeze(0)

            # Transform the default points by the end effector pose
            pc = transform_pointcloud(
                default_points.repeat(poses.size(0), 1, 1),
                poses,
                in_place=True,
            )

        # If num_points is not provided, use self.num_fixed_points
        if num_points is None:
            num_points = self.num_fixed_points

        if num_points is None:
            return pc

        # If we request more points than available, duplicate some points
        if num_points > pc.shape[1]:
            # Calculate how many times we need to repeat the points
            repeat_factor = int(np.ceil(num_points / pc.shape[1]))
            # Repeat the points
            repeated_pc = pc.repeat_interleave(repeat_factor, dim=1)
            # Take exactly num_points
            result = repeated_pc[:, :num_points, :]
            return result
        else:
            # Sample without replacement if we have enough points
            indices = np.random.choice(pc.shape[1], num_points, replace=False)
            result = pc[:, indices, :]
            return result

    def sample(self, config, num_points=None, return_distribution=False):
        """
        Samples points from the surface of the robot by calling fk.

        Parameters
        ----------
        config : Tensor of length (M,) or (N, M) where M is the number of
                 actuated joints. N is the batch size.
        num_points : Number of points desired. If None, uses self.num_fixed_points.
        return_distribution : If True, returns tuple of (pointcloud, sampling_distribution).
                             If False, returns only pointcloud (backward compatible).

        Returns
        -------
        If return_distribution=False: N x num points x 3 pointcloud of robot points
        If return_distribution=True: Tuple of (pointcloud, sampling_distribution):
        - pointcloud: N x num points x 3 pointcloud of robot points
        - sampling_distribution: N x num_links tensor indicating number of points sampled from each link
        """
        # If deterministic sampling is enabled, create a new RNG instance with the same seed
        if self.deterministic_sampling:
            self.rng = np.random.RandomState(self._rng_seed)

        # If num_points is not provided, use self.num_fixed_points
        if num_points is None:
            num_points = self.num_fixed_points

        if config.ndim == 1:
            config = config.unsqueeze(0)

        # Handle prismatic joints if needed
        if hasattr(self.robot, "prismatic_joints") and self.robot.prismatic_joints:
            # Add default values for prismatic joints
            cfg = torch.cat(
                (
                    config,
                    self.default_prismatic_value
                    * torch.ones(
                        (config.shape[0], len(self.robot.prismatic_joints)),
                        device=config.device,
                    ),
                ),
                dim=1,
            )
        else:
            cfg = config

        fk = self.torch_robot.visual_geometry_fk_batch(cfg)
        values = list(fk.values())
        assert len(self.links) == len(values)

        fk_transforms = {}
        fk_points = []

        for idx, l in enumerate(self.links):
            # Skip base link if requested
            if l.name == self.robot.link_names[0] and not self.with_base_link:
                continue

            fk_transforms[l.name] = values[idx]
            # repeat points across batch dimension in case fk transforms are batched
            original_pc = (
                self.points[l.name]
                .float()
                .repeat((fk_transforms[l.name].shape[0], 1, 1))
            )
            pc = transform_pointcloud(
                original_pc,
                fk_transforms[l.name],
                in_place=True,
            )
            fk_points.append(pc)

        # collate all points across links in one tensor
        pc = torch.cat(fk_points, dim=1)

        if num_points is None:
            # Return full point cloud with distribution showing all available points
            sampling_distribution = (
                torch.tensor(
                    [points.shape[1] for points in fk_points], device=config.device
                )
                .unsqueeze(0)
                .expand(config.shape[0], -1)
            )
            return (pc, sampling_distribution) if return_distribution else pc

        # Calculate points per link based on their original proportions
        total_points = pc.shape[1]
        points_per_link = []
        start_idx = 0
        sampling_distribution = []

        # Use original proportional allocation
        for i, points in enumerate(fk_points):
            link_points = points.shape[1]
            proportion = link_points / total_points
            points_to_sample = int(np.round(num_points * proportion))
            points_per_link.append(
                (start_idx, start_idx + link_points, points_to_sample)
            )
            sampling_distribution.append(points_to_sample)
            start_idx += link_points

        # Sample points from each link segment
        sampled_points = []
        total_sampled = 0
        for start, end, n_points in points_per_link:
            segment = pc[:, start:end, :]
            if n_points > (end - start):
                # If we need more points than available, duplicate some
                repeat_factor = int(np.ceil(n_points / (end - start)))
                repeated = segment.repeat_interleave(repeat_factor, dim=1)
                if self.deterministic_sampling:
                    indices = self.rng.choice(
                        repeated.shape[1], n_points, replace=False
                    )
                else:
                    indices = np.random.choice(
                        repeated.shape[1], n_points, replace=False
                    )
                sampled = repeated[:, indices, :]
            else:
                if self.deterministic_sampling:
                    indices = self.rng.choice(end - start, n_points, replace=False)
                else:
                    indices = np.random.choice(end - start, n_points, replace=False)
                sampled = segment[:, indices, :]
            sampled_points.append(sampled)
            total_sampled += n_points

        # Concatenate all sampled points
        result = torch.cat(sampled_points, dim=1)

        # Ensure we return exactly the requested number of points
        # Due to rounding errors when allocating points across links,
        # there's a chance we might not get exactly num_points.
        # This fix ensures we always return exactly num_points by either
        # duplicating points (if we have too few) or truncating (if we have too many).
        # This is critical for downstream operations expecting a specific tensor size.
        if result.shape[1] != num_points:
            if result.shape[1] < num_points:
                # If we have fewer points than requested, duplicate some
                missing = num_points - result.shape[1]
                # Randomly select indices to duplicate
                if self.deterministic_sampling:
                    indices = self.rng.choice(result.shape[1], missing, replace=True)
                else:
                    indices = np.random.choice(result.shape[1], missing, replace=True)
                duplicated = result[:, indices, :]
                result = torch.cat([result, duplicated], dim=1)
            else:
                # If we have more points than requested, truncate
                result = result[:, :num_points, :]

        # Convert sampling distribution to tensor and expand to batch dimension
        sampling_distribution = torch.tensor(
            sampling_distribution, device=config.device
        )
        sampling_distribution = sampling_distribution.unsqueeze(0).expand(
            config.shape[0], -1
        )

        return (result, sampling_distribution) if return_distribution else result


class FrankaCollisionSampler:
    def __init__(
        self,
        device,
        default_prismatic_value=0.025,
        with_base_link=True,
        margin=0.0,
    ):
        logging.getLogger("trimesh").setLevel("ERROR")
        self.default_prismatic_value = default_prismatic_value
        self.robot = TorchURDF.load(
            FrankaRobot.urdf, lazy_load_meshes=True, device=device
        )
        self.spheres = []
        for radius, point_set in FrankaRobot.SPHERES:
            sphere_centers = {
                k: torch.as_tensor(v).to(device) for k, v in point_set.items()
            }
            if not with_base_link:
                sphere_centers = {
                    k: v for k, v in sphere_centers.items() if k != "panda_link0"
                }
            if not len(sphere_centers):
                continue
            self.spheres.append(
                (
                    radius + margin,
                    sphere_centers,
                )
            )

        all_spheres = {}
        for radius, point_set in FrankaRobot.SPHERES:
            for link_name, centers in point_set.items():
                if not with_base_link and link_name == "panda_link0":
                    continue
                for c in centers:
                    all_spheres[link_name] = all_spheres.get(link_name, []) + [
                        Sphere(c, radius + margin)
                    ]

        total_points = 10000
        surface_scalar_sum = sum(
            [sum([s.radius**2 for s in v]) for v in all_spheres.values()]
        )
        surface_scalar = total_points / surface_scalar_sum
        self.link_points = {}
        for link_name, spheres in all_spheres.items():
            self.link_points[link_name] = torch.as_tensor(
                np.concatenate(
                    [
                        s.sample_surface(int(surface_scalar * s.radius**2))
                        for s in spheres
                    ],
                    axis=0,
                ),
                device=device,
            )

    def sample(self, config, n):
        if config.ndim == 1:
            config = config.unsqueeze(0)
        cfg = torch.cat(
            (
                config,
                self.default_prismatic_value
                * torch.ones((config.shape[0], 2), device=config.device),
            ),
            dim=1,
        )
        fk = self.robot.link_fk_batch(cfg, use_names=True)
        pointcloud = []
        for link_name, points in self.link_points.items():
            pc = transform_pointcloud(
                points.float().repeat((fk[link_name].shape[0], 1, 1)),
                fk[link_name],
                in_place=True,
            )
            pointcloud.append(pc)
        pc = torch.cat(pointcloud, dim=1)
        return pc[:, np.random.choice(pc.shape[1], n, replace=False), :]

    def compute_spheres(self, config):
        if config.ndim == 1:
            config = config.unsqueeze(0)
        cfg = torch.cat(
            (
                config,
                self.default_prismatic_value
                * torch.ones((config.shape[0], 2), device=config.device),
            ),
            dim=1,
        )
        fk = self.robot.link_fk_batch(cfg, use_names=True)
        points = []
        for radius, spheres in self.spheres:
            fk_points = []
            for link_name in spheres:
                pc = transform_pointcloud(
                    spheres[link_name]
                    .type_as(cfg)
                    .repeat((fk[link_name].shape[0], 1, 1)),
                    fk[link_name].type_as(cfg),
                    in_place=True,
                )
                fk_points.append(pc)
            points.append((radius, torch.cat(fk_points, dim=1)))
        return points


class FrankaSelfCollisionSampler(NumpySelfCollisionSampler):
    def __init__(self, device, default_prismatic_value=0.025):
        super().__init__(default_prismatic_value)
        self.robot = TorchURDF.load(
            FrankaRobot.urdf, lazy_load_meshes=True, device=device
        )
        for k, v in self.link_points.items():
            self.link_points[k] = torch.as_tensor(v, device=device).unsqueeze(0)

    def sample(self, config, n):
        if config.ndim == 1:
            config = config.unsqueeze(0)
        cfg = torch.cat(
            (
                config,
                self.default_prismatic_value
                * torch.ones((config.shape[0], 2), device=config.device),
            ),
            dim=1,
        )
        fk = self.robot.link_fk_batch(cfg, use_names=True)
        pointcloud = []
        for link_name, points in self.link_points.items():
            pc = transform_pointcloud(
                points.float().repeat((fk[link_name].shape[0], 1, 1)),
                fk[link_name],
                in_place=True,
            )
            pointcloud.append(pc)
        pc = torch.cat(pointcloud, dim=1)
        return pc[:, np.random.choice(pc.shape[1], n, replace=False), :]
