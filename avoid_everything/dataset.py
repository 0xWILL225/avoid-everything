import hashlib
import logging
from dataclasses import dataclass
from functools import wraps

import h5py
import numpy as np
from geometrout.primitive import Cuboid, CuboidArray, Cylinder, CylinderArray
from robofin.robots import FrankaRobot
from tqdm.auto import tqdm, trange

from avoid_everything.type_defs import PlanningProblem


def chunk_indices(indices, chunk_size):
    nchunks = len(indices) // chunk_size
    chunks = [
        np.asarray(indices[i * chunk_size : (i + 1) * chunk_size])
        for i in range(nchunks + 1)
    ]
    chunks = [c for c in chunks if len(c) > 0]
    return chunks


@dataclass
class FlattenedObstacles:
    cuboid_centers: np.ndarray
    cuboid_dims: np.ndarray
    cuboid_quaternions: np.ndarray
    cylinder_centers: np.ndarray
    cylinder_radii: np.ndarray
    cylinder_heights: np.ndarray
    cylinder_quaternions: np.ndarray

    def hashable(self, precision=6):
        return tuple(
            (
                np.concatenate(
                    (
                        self.cuboid_centers.flatten(),
                        self.cuboid_dims.flatten(),
                        self.cuboid_quaternions.flatten(),
                        self.cylinder_centers.flatten(),
                        self.cylinder_radii.flatten(),
                        self.cylinder_heights.flatten(),
                        self.cylinder_quaternions.flatten(),
                    )
                )
                * pow(10, precision)
            )
            .astype(int)
            .tolist()
        )


def check_file(fn):
    @wraps(fn)
    def wrapper(self, *args, **kw):
        assert self.file.id, "HDF5 file must be open to access data"
        return fn(self, *args, **kw)

    return wrapper


INDEX_SPECIAL_TOKEN = "_index"
LENGTHS_SPECIAL_TOKEN = "_lengths"
PERSISTENT_KEY = "cuboid_centers"  # An always present key that has the length of the number of problems
WELL_INDEXED = "well_indexed"


def _index(key):
    return f"{key}{INDEX_SPECIAL_TOKEN}"


def _pindex(key):
    return f"{key}{LENGTHS_SPECIAL_TOKEN}"


class UnindexedKeyedData:
    """
    Only needed for datasets that are in process of being built.
    Otherwise use KeyedData
    """

    def __init__(self, key, file, mode="r"):
        self.file = file
        assert self.file.id, "HDF5 file must be open to access data"
        self.key = key
        self.mode = mode
        self.guard_pindex()

    @check_file
    def __len__(self):
        raise NotImplementedError("Cannot compute length for UnindexedKeyedData")

    @property
    def well_indexed(self):
        return self.file[self.key].attrs[WELL_INDEXED]

    @property
    def pindex(self):
        return _pindex(self.key)

    @property
    def index(self):
        """
        Using a nonsensical key to prevent mixups
        """
        return _index(self.key)

    @check_file
    def get_expert_indices(self):
        """
        Returns the indices where this expert has solutions.
        Useful for filtering out unsolved problems.
        """
        return np.flatnonzero(self.file[self.pindex])

    @check_file
    def expert_length(self, pidx):
        return self.file[self.pindex][pidx]

    @check_file
    def max_expert_length(self):
        return self.file[self.key].shape[1]

    @check_file
    def set_expert(self, pidx, expert):
        assert self.mode in ("w", "w-", "r+")
        T = len(expert)
        assert T <= self.max_expert_length()
        self.file[self.key][pidx, :] = np.pad(
            expert, ((0, self.max_expert_length() - T), (0, 0)), mode="edge"
        )
        if T != self.expert_length(pidx):
            self.file[self.pindex][pidx] = T
            self.file[self.key].attrs[WELL_INDEXED] = False

    @check_file
    def guard_pindex(self):
        if self.pindex not in self.file.keys():
            raise KeyError(f"{self.pindex} not found in dataset")

    @check_file
    def problem(self, pidx):
        obstacles = []
        if "cuboid_centers" in self.file.keys():
            for c, d, q in zip(
                self.file["cuboid_centers"][pidx],
                self.file["cuboid_dims"][pidx],
                self.file["cuboid_quaternions"][pidx],
            ):
                cub = Cuboid(c, d, q)
                if not cub.is_zero_volume():
                    obstacles.append(cub)
        if "cylinder_centers" in self.file.keys():
            for c, r, h, q in zip(
                self.file["cylinder_centers"][pidx],
                self.file["cylinder_radii"][pidx],
                self.file["cylinder_heights"][pidx],
                self.file["cylinder_quaternions"][pidx],
            ):
                cyl = Cylinder(c, r.item(), h.item(), q)
                if not cyl.is_zero_volume():
                    obstacles.append(cyl)

        return PlanningProblem(
            target=self.file[self.key][pidx, -1],
            q0=self.file[self.key][pidx, 0],
            obstacles=obstacles,
        )

    @check_file
    def robometrics_problem(self, pidx, world_frame, eff_frame):
        problem = self.problem(pidx)
        goal_pose = FrankaRobot.fk(problem.target, eff_frame=eff_frame)
        cuboids = []
        cylinders = []
        assert problem.obstacles is not None
        for o in problem.cuboids:
            cuboids.append(
                {
                    "dims": [float(x) for x in o.dims],
                    "pose": [float(x) for x in [*o.pose.xyz, *o.pose.so3.wxyz]],
                }
            )
        for o in problem.cylinders:
            cylinders.append(
                {
                    "height": float(o.height),
                    "radius": float(o.radius),
                    "pose": [float(x) for x in [*o.pose.xyz, *o.pose.so3.wxyz]],
                }
            )
        obstacles = {
            "cuboids": {f"cuboid{i}": cuboid for i, cuboid in enumerate(cuboids)},
            "cylinders": {
                f"cylinder{i}": cylinder for i, cylinder in enumerate(cylinders)
            },
        }

        assert isinstance(problem.target, np.ndarray)

        return {
            "collision_buffer_ik": 0.0,
            "goal_ik": [float(x) for x in problem.target.tolist()],
            "goal_pose": {
                "frame": eff_frame,
                "position_xyz": [float(x) for x in goal_pose.xyz],
                "quaternion_wxyz": [float(x) for x in goal_pose.so3.wxyz],
            },
            "world_frame": world_frame,
            "start": [float(x) for x in problem.q0.tolist()],
            "obstacles": obstacles,
        }

    @check_file
    def primitive_arrays(self, pidx):
        ret = []
        if "cuboid_centers" in self.file.keys():
            cuboids = []
            for c, d, q in zip(
                self.file["cuboid_centers"][pidx],
                self.file["cuboid_dims"][pidx],
                self.file["cuboid_quaternions"][pidx],
            ):
                cub = Cuboid(c, d, q)
                if not cub.is_zero_volume():
                    cuboids.append(cub)
            if cuboids:
                ret.append(CuboidArray(cuboids))
        if "cylinder_centers" in self.file.keys():
            cylinders = []
            for c, r, h, q in zip(
                self.file["cylinder_centers"][pidx],
                self.file["cylinder_radii"][pidx],
                self.file["cylinder_heights"][pidx],
                self.file["cylinder_quaternions"][pidx],
            ):
                cyl = Cylinder(c, r.squeeze(), h.squeeze(), q)
                if not cyl.is_zero_volume():
                    cylinders.append(cyl)
            if cylinders:
                ret.append(CylinderArray(cylinders))
        return ret

    @check_file
    def flattened_obstacles(self, pidx):
        if "cuboid_centers" in self.file.keys():
            cuboid_dims = self.file["cuboid_dims"][pidx]
            if cuboid_dims.ndim == 1:
                cuboid_dims = np.expand_dims(cuboid_dims, axis=0)

            cuboid_centers = self.file["cuboid_centers"][pidx]
            if cuboid_centers.ndim == 1:
                cuboid_centers = np.expand_dims(cuboid_centers, axis=0)

            cuboid_quaternions = self.file["cuboid_quaternions"][pidx]
            if cuboid_quaternions.ndim == 1:
                cuboid_quaternions = np.expand_dims(cuboid_quaternions, axis=0)
            # Entries without a shape are stored with an invalid quaternion of all zeros
            # This will cause NaNs later in the pipeline. It's best to set these to unit
            # quaternions.
            # To find invalid shapes, we just look for a dimension with size 0
            cuboid_quaternions[np.all(np.isclose(cuboid_quaternions, 0), axis=1), 0] = 1
        else:
            # Create a dummy cuboid if cylinders aren't in the hdf5 file
            cuboid_centers = np.array([[0.0, 0.0, 0.0]])
            cuboid_dims = np.array([[0.0, 0.0, 0.0]])
            cuboid_quaternions = np.array([[1.0, 0.0, 0.0, 0.0]])

        if "cylinder_centers" in self.file.keys():
            cylinder_radii = self.file["cylinder_radii"][pidx]
            if cylinder_radii.ndim == 1:
                cylinder_radii = np.expand_dims(cylinder_radii, axis=0)
            cylinder_heights = self.file["cylinder_heights"][pidx]
            if cylinder_heights.ndim == 1:
                cylinder_heights = np.expand_dims(cylinder_heights, axis=0)
            cylinder_centers = self.file["cylinder_centers"][pidx]
            if cylinder_centers.ndim == 1:
                cylinder_centers = np.expand_dims(cylinder_centers, axis=0)
            cylinder_quaternions = self.file["cylinder_quaternions"][pidx]
            if cylinder_quaternions.ndim == 1:
                cylinder_quaternions = np.expand_dims(cylinder_quaternions, axis=0)
            # Ditto to the comment above about fixing ill-formed quaternions
            cylinder_quaternions[
                np.all(np.isclose(cylinder_quaternions, 0), axis=1), 0
            ] = 1
        else:
            # Create a dummy cylinder if cylinders aren't in the hdf5 file
            cylinder_radii = np.array([[0.0]])
            cylinder_heights = np.array([[0.0]])
            cylinder_centers = np.array([[0.0, 0.0, 0.0]])
            cylinder_quaternions = np.array([[1.0, 0.0, 0.0, 0.0]])
        return FlattenedObstacles(
            cuboid_centers,
            cuboid_dims,
            cuboid_quaternions,
            cylinder_centers,
            cylinder_radii,
            cylinder_heights,
            cylinder_quaternions,
        )

    @check_file
    def scene_hash(self, pidx, precision=6):
        flobs = self.flattened_obstacles(pidx)
        return flobs.hashable(precision)

    @check_file
    def one_pidx_per_scene(self, precision=6):
        """Returns indices of problems with no scene duplicates."""
        all_indices = self.get_expert_indices()
        scenes = set()
        pidxs = []
        for pidx in tqdm(all_indices, desc="Hashing scenes"):
            key = self.scene_hash(pidx, precision)
            if key not in scenes:
                scenes.add(key)
                pidxs.append(pidx)
        return pidxs

    @check_file
    def partition_by_scene(self, approx_subset_size, precision=6):
        """Partitions into two subsets with no overlapping scenes."""
        all_indices = self.get_expert_indices()
        assert approx_subset_size <= len(all_indices)
        scenes = {}
        for pidx in tqdm(all_indices, desc="Hashing scenes"):
            key = self.scene_hash(pidx, precision)
            scenes[key] = scenes.get(key, []) + [pidx]
        chunk1, chunk2 = [], []
        for key, pidxs in tqdm(scenes.items(), desc="Splitting"):
            if len(chunk1) <= approx_subset_size:
                chunk1.extend(pidxs)
            else:
                chunk2.extend(pidxs)

        return chunk1, chunk2

    @check_file
    def padded_expert(self, pidx):
        H = self.file[self.key].shape[1]
        T = self.expert_length(pidx)
        return np.pad(self.file[self.key][pidx, :T], ((0, H - T), (0, 0)), mode="edge")

    @check_file
    def expert(self, pidx):
        T = self.expert_length(pidx)
        return self.file[self.key][pidx, :T]

    @check_file
    def stats(self, num_trajectory_samples=None):
        N = self.file[self.key].shape[0]
        if num_trajectory_samples is not None:
            indices = sorted(np.random.permutation(N)[:num_trajectory_samples])
        else:
            indices = np.arange(N)
        chunks = chunk_indices(indices, 10000)
        n = 0
        mean = np.zeros(self.file[self.key].shape[-1])
        M2 = 0
        for chunk in tqdm(chunks, desc="Computing dataset stats"):
            batch = self.file[self.key][chunk, ...]
            lengths = self.file[self.pindex][chunk]
            mask = np.arange(batch.shape[1]) < lengths[:, None]
            batch = batch[mask]
            n += len(batch)
            delta = batch - mean
            mean += np.sum(delta / n, axis=0)
            delta2 = batch - mean
            M2 += np.sum(delta * delta2, axis=0)

        return {"mean": mean, "variance": M2 / n if n > 1 else 0}


class KeyedData(UnindexedKeyedData):
    def __init__(self, key, file, mode="r"):
        super().__init__(key, file, mode)
        self.guard_index()

    @check_file
    def lookup_pidx(self, sidx):
        pidx, _ = self.file[self.index][sidx]
        return pidx

    @check_file
    def lookup_start_sidx(self, pidx):
        left, right = 0, len(self) - 1

        while left <= right:
            mid = (left + right) // 2
            mid_pidx = self.lookup_pidx(mid)

            if mid_pidx == pidx:
                # Since we want the first occurrence of pidx,
                # continue to search in the left half even if we've found pidx.
                if mid == 0 or self.lookup_pidx(mid - 1) != pidx:
                    return mid
                right = mid - 1
            elif mid_pidx < pidx:
                left = mid + 1
            else:
                right = mid - 1

        return None

    @check_file
    def how_far_along_expert(self, sidx):
        pidx, ts = self.file[self.index][sidx]
        el = self.expert_length(pidx)
        if el <= 1:
            return 1.0
        return float(ts / (el - 1.0))

    @check_file
    def state_range(self, sidx, lookahead):
        pidx, ts = self.file[self.index][sidx]
        x = self.file[self.key][pidx, ts : ts + lookahead]
        if x.shape[0] < lookahead:
            pad = lookahead - x.shape[0]
            x = np.pad(x, ((0, pad), (0, 0)), mode="edge")
        return x

    @check_file
    def guard_index(self):
        if self.index not in self.file.keys() and self.file[self.key][WELL_INDEXED]:
            raise KeyError(f"{self.index} not found in dataset")

    @check_file
    def state(self, sidx):
        pidx, ts = self.file[self.index][sidx]
        return self.file[self.key][pidx, ts]

    @check_file
    def state_action(self, sidx):
        pidx, ts = self.file[self.index][sidx]
        xt = self.file[self.key][pidx, ts]
        if ts == self.file[self.key].shape[1] - 1:
            dxt = np.zeros_like(xt)
        else:
            dxt = self.file[self.key][pidx, ts + 1] - xt
        return xt, dxt

    @check_file
    def rl_info(self, sidx):
        """
        Returns state, action, next state, and whether run was terminated
        """
        pidx, ts = self.file[self.index][sidx]
        xt = self.file[self.key][pidx, ts]
        if ts == self.file[self.key].shape[1] - 1:
            return xt, np.zeros_like(xt), xt, 1
        dxt = self.file[self.key][pidx, ts + 1] - xt
        return xt, dxt, self.file[self.key][pidx, ts + 1], 0

    @check_file
    def state_state(self, sidx):
        pidx, ts = self.file[self.index][sidx]
        if ts == self.file[self.key].shape[1] - 1:
            xt = self.file[self.key][pidx, ts][None, :]
            return xt.repeat(2, axis=0)
        return self.file[self.key][pidx, ts : ts + 2]

    @check_file
    def __len__(self):
        return len(self.file[self.index])


class Dataset:
    def __init__(self, file_path, mode="r"):
        self.file_path = file_path
        self.file = h5py.File(str(self.file_path), mode)
        self.keys = list(self.file.keys())
        self.mode = mode

    @property
    def md5_checksum(self):
        with open(self.file_path, "rb") as f:
            file_hash = hashlib.md5()
            while chunk := f.read(8192):
                file_hash.update(chunk)
        return file_hash.hexdigest()

    @check_file
    def __len__(self):
        return len(self.file[PERSISTENT_KEY])

    @check_file
    def rebuild_index(self, key):
        N = 0
        keyed_data = UnindexedKeyedData(key, self.file, self.mode)
        for pidx in trange(len(self), desc="Counting states"):
            N += keyed_data.expert_length(pidx)
        index = np.zeros((N, 2), dtype=int)
        total = 0
        for pidx in trange(len(self), desc="Assembling starts and ends"):
            el = keyed_data.expert_length(pidx)
            index[total : total + el, 0] = pidx
            index[total : total + el, 1] = np.arange(el)
            total += el
        if self.mode in ("w", "w-", "r+"):
            if keyed_data.index in self.file.keys():
                del self.file[keyed_data.index]
            ds = self.file.create_dataset(keyed_data.index, index.shape, dtype=int)
            ds[...] = index
            self.file[key].attrs[WELL_INDEXED] = True
            logging.info(f"Done writing index for {key}")
        else:
            logging.warning(f"Index not written as dataset mode is {self.mode}")
        return index

    @classmethod
    def merge(
        cls,
        destination,
        paths_to_merge,
        rebuild_index=True,
        mode="w-",
        skip_extra_keys=False,  # Skips keys not shared by all datasets
    ):
        assert isinstance(paths_to_merge, list)
        # First check that they all have the same keys and count total
        keys = None
        keys_to_index = set()
        N = 0
        logging.info("Checking datasets")
        for p in tqdm(paths_to_merge, desc="Assembling keys"):
            with Dataset(p) as f:
                if keys is None:
                    keys = set(
                        [key for key in f.keys if INDEX_SPECIAL_TOKEN not in key]
                    )
                fkeys = set([key for key in f.keys if INDEX_SPECIAL_TOKEN not in key])
                if skip_extra_keys:
                    keys_to_skip = keys.difference(fkeys).union(fkeys.difference(keys))
                    keys = keys.intersection(fkeys)
                    logging.warn(f"Skipping keys: {keys_to_skip}")
                else:
                    assert keys == fkeys, "Not all datasets have the same keys"
                keys_to_index.update(
                    [
                        k.replace(INDEX_SPECIAL_TOKEN, "")
                        for k in f.keys
                        if INDEX_SPECIAL_TOKEN in k
                    ]
                )
                N += len(f)
        assert keys is not None
        logging.info("Merging")
        with h5py.File(destination, mode) as g:
            gsf = 0
            for ii, p in enumerate(tqdm(paths_to_merge, desc="Merging")):
                with h5py.File(p) as f:
                    n = len(f[PERSISTENT_KEY])  # To get the
                    chunks = chunk_indices(np.arange(n), 10000)
                    for key in keys:
                        if ii == 0:
                            ds = g.create_dataset(
                                key, (N, *f[key].shape[1:]), dtype=f[key].dtype
                            )
                        else:
                            ds = g[key]
                        fsf = 0
                        for chunk in chunks:
                            ds[gsf + fsf : gsf + fsf + len(chunk), ...] = f[key][
                                chunk, ...
                            ]
                            fsf += len(chunk)
                    gsf += n
        with Dataset(destination, "r+") as g:
            for key in keys_to_index:
                if rebuild_index:
                    g.rebuild_index(key)
                else:
                    g.file[key].attrs[WELL_INDEXED] = False

    @classmethod
    def merge_with_unequal_sizes(
        cls,
        destination,
        paths_to_merge,
        rebuild_index=True,
        mode="w-",
        skip_extra_keys=False,  # Skips keys not shared by all datasets
    ):
        assert isinstance(paths_to_merge, list)
        # First check that they all have the same keys and count total
        keys = None
        keys_to_index = set()
        N = 0
        logging.info("Checking datasets")
        shapes = None
        for p in tqdm(paths_to_merge, desc="Assembling keys"):
            with Dataset(p) as f:
                if keys is None:
                    keys = set(
                        [key for key in f.keys if INDEX_SPECIAL_TOKEN not in key]
                    )
                fkeys = set([key for key in f.keys if INDEX_SPECIAL_TOKEN not in key])
                if skip_extra_keys:
                    keys_to_skip = keys.difference(fkeys).union(fkeys.difference(keys))
                    keys = keys.intersection(fkeys)
                    logging.warn(f"Skipping keys: {keys_to_skip}")
                else:
                    diff = (set(keys) - set(fkeys)).union(set(fkeys) - set(keys))
                    assert (
                        keys == fkeys
                    ), f"Not all datasets have the same keys ({diff}). Use skip_extra_keys to skip these."
                keys_to_index.update(
                    [
                        k.replace(INDEX_SPECIAL_TOKEN, "")
                        for k in f.keys
                        if INDEX_SPECIAL_TOKEN in k
                    ]
                )
                N += len(f)
        assert keys is not None
        shapes = {}
        for p in tqdm(paths_to_merge, desc="Assembling shapes"):
            with h5py.File(str(p)) as f:
                for key in f.keys():
                    if key in shapes and isinstance(shapes[key], list):
                        shapes[key] = shapes[key] + [f[key].shape[1:]]
                    elif key in shapes and shapes[key] != f[key].shape[1:]:
                        shapes[key] = [shapes[key], f[key].shape[1:]]
                    elif key in shapes:
                        pass
                    else:
                        shapes[key] = f[key].shape[1:]
        multishapes = {key: val for key, val in shapes.items() if isinstance(val, list)}
        shapes = {
            **{k: tuple((max(*x) for x in zip(*v))) for k, v in multishapes.items()},
            **{k: v for k, v in shapes.items() if not isinstance(v, list)},
        }

        with h5py.File(destination, mode) as g:
            gsf = 0
            for ii, p in enumerate(tqdm(paths_to_merge, desc="Merging")):
                with h5py.File(p) as f:
                    n = len(f[PERSISTENT_KEY])  # To get the
                    chunks = chunk_indices(np.arange(n), 10000)
                    for key in keys:
                        if ii == 0:
                            ds = g.create_dataset(
                                key, (N, *shapes[key]), dtype=f[key].dtype
                            )
                        else:
                            ds = g[key]
                        fsf = 0
                        slices = tuple((slice(x) for x in f[key].shape[1:]))
                        for chunk in chunks:
                            if len(slices):
                                assign_to = tuple(
                                    [
                                        slice(gsf + fsf, gsf + fsf + len(chunk)),
                                        *slices,
                                    ]
                                )
                                assign_from = tuple([chunk, *slices])
                            else:
                                assign_to = slice(gsf + fsf, gsf + fsf + len(chunk))
                                assign_from = chunk
                            ds[assign_to] = f[key][assign_from]
                            fsf += len(chunk)
                    gsf += n
        with Dataset(destination, "r+") as g:
            for key in keys_to_index:
                if rebuild_index:
                    g.rebuild_index(key)
                else:
                    g.file[key].attrs[WELL_INDEXED] = False

    @check_file
    def copy(self, path, indices, rebuild_index=True, mode="w-"):
        assert len(indices) == len(np.unique(indices)), "Indices must be unique"
        logging.info("Sorting indices")
        indices = sorted(indices)
        index_chunks = chunk_indices(indices, 10000)
        index_keys = []
        with h5py.File(path, mode) as g:
            for key in tqdm(self.file.keys(), desc="Copying data"):
                if INDEX_SPECIAL_TOKEN in key:
                    logging.info(f"Skipping index: {key}")
                    index_keys.append(key.replace(INDEX_SPECIAL_TOKEN, ""))
                else:
                    ds = g.create_dataset(
                        key,
                        (len(indices), *self.file[key].shape[1:]),
                        dtype=self.file[key].dtype,
                    )
                    sf = 0
                    for chunk in index_chunks:
                        ds[sf : sf + len(chunk), ...] = self.file[key][chunk, ...]
                        sf += len(chunk)
            with Dataset(path, "r+") as g:
                for key in index_keys:
                    if rebuild_index:
                        logging.info(f"Rebuilding index for {key}")
                        g.rebuild_index(key)
                    else:
                        g.file[key].attrs[WELL_INDEXED] = False

    @check_file
    def copy_with_dupicates(self, path, indices, rebuild_index=True, mode="w-"):
        logging.info("Sorting indices with duplicates allowed")
        indices = sorted(indices)

        index_chunks = chunk_indices(indices, 10000)
        index_keys = []
        with h5py.File(path, mode) as g:
            for key in tqdm(self.file.keys(), desc="Copying data"):
                if INDEX_SPECIAL_TOKEN in key:
                    logging.info(f"Skipping index: {key}")
                    index_keys.append(key.replace(INDEX_SPECIAL_TOKEN, ""))
                else:
                    ds = g.create_dataset(
                        key,
                        (len(indices), *self.file[key].shape[1:]),
                        dtype=self.file[key].dtype,
                    )
                    sf = 0
                    for chunk in index_chunks:
                        unique_indices, counts = np.unique(chunk, return_counts=True)
                        data_chunk = self.file[key][unique_indices, ...]
                        ds[sf : sf + len(chunk), ...] = np.repeat(
                            data_chunk, counts, axis=0
                        )
                        sf += len(chunk)
            with Dataset(path, "r+") as g:
                for key in index_keys:
                    if rebuild_index:
                        logging.info(f"Rebuilding index for {key}")
                        g.rebuild_index(key)
                    else:
                        g.file[key].attrs[WELL_INDEXED] = False

    def add_column(self, column_name, data):
        if PERSISTENT_KEY in self.file.keys():
            assert len(data) == len(self)
        else:
            assert column_name == PERSISTENT_KEY
        new_column = self.file.create_dataset(column_name, data.shape, dtype=data.dtype)

        indices = np.arange(len(data))
        nchunks = len(indices) // 10000
        index_chunks = [
            np.asarray(indices[i * 10000 : (i + 1) * 10000]) for i in range(nchunks + 1)
        ]
        # If the number of indices is 0 mod 10000, there's an empty chunk at the end
        index_chunks = [c for c in index_chunks if len(c) > 0]
        sf = 0
        for chunk in index_chunks:
            new_column[sf : sf + len(chunk), ...] = data[chunk, ...]
            sf += len(chunk)

    def add_expert(self, expert_name, expert_data, expert_lengths, build_index=True):
        assert self.mode in ("w", "w-", "r+")
        expert = self.file.create_dataset(
            expert_name, expert_data.shape, expert_data.dtype
        )
        pindices = self.file.create_dataset(
            _pindex(expert_name), expert_lengths.shape, dtype=int
        )
        indices = np.arange(len(expert_data))
        nchunks = len(indices) // 10000
        index_chunks = [
            np.asarray(indices[i * 10000 : (i + 1) * 10000]) for i in range(nchunks + 1)
        ]
        # If the number of indices is 0 mod 10000, there's an empty chunk at the end
        index_chunks = [c for c in index_chunks if len(c) > 0]
        sf = 0
        for chunk in index_chunks:
            expert[sf : sf + len(chunk), ...] = expert_data[chunk, ...]
            pindices[sf : sf + len(chunk), ...] = expert_lengths[chunk, ...]
            sf += len(chunk)
        if build_index:
            self.rebuild_index(expert_name)
        else:
            expert.attrs[WELL_INDEXED] = False

    def delete_expert(self, expert_name):
        index = _index(expert_name)
        if index in self.file.keys():
            logging.info(f"Deleting index: {index}")
            del self.file[index]
        else:
            logging.info(f"Index {index} not found")
        pindex = _pindex(expert_name)
        if pindex in self.file.keys():
            logging.info(f"Deleting p-index: {pindex}")
            del self.file[pindex]
        else:
            logging.info(f"P-Index {pindex} not found")
        if expert_name in self.file.keys():
            logging.info(f"Deleting expert {expert_name}")
            del self.file[expert_name]
        else:
            logging.info(f"Expert {expert_name} not found")

    def _ipython_key_completions_(self):
        """
        This isn't going to be totally accuracte because
        the keys will reflect the last time this file was open. If
        you make multiple objects with different paths and the same
        variable name, you'll get wonky autocomplete
        """
        return self.keys

    @check_file
    def __getitem__(self, key):
        assert key in self.keys, f"Dataset {key} not among {list(self.keys)}"
        if WELL_INDEXED not in self.file[key].attrs:
            logging.warning(f"Must rebuild index for {key} to maintain trustworthiness")
        if not self.file[key].attrs.get(WELL_INDEXED, False):
            return UnindexedKeyedData(key, self.file, self.mode)
        return KeyedData(key, self.file, self.mode)

    def close(self):
        if self.file.id:
            self.file.close()

    def __enter__(self):
        # This is a no-op but kept in for best-practice
        self.file.__enter__()
        return self

    def __exit__(self, *args):
        # Closes the file
        self.file.__exit__(*args)
