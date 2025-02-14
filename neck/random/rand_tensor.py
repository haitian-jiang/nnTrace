import numpy as np
import torch
import xxhash
from typing import List, Tuple, Optional
from .random_backend import seed_to_rand_parallel
# from numba import njit


def _slice_to_index(slices: Tuple[slice]) -> Tuple[np.ndarray]:
    """Convert a tuple of slices to a tuple of indices.
    
    Args:
        slices (Tuple[slice]): A tuple of slice objects.
    
    Returns:
        Tuple[np.ndarray]: A tuple of numpy arrays containing the indices for each slice.
    """
    # Do not support reverse or strided slices now.
    for s in slices:
        assert s.stop > s.start, f"Slice {s} is invalid."
        assert s.step is None or s.step == 1, f"Slice {s} is invalid."
    # Determine the shape of the slice
    slice_shape = tuple((s.stop - s.start) for s in slices)
    # Generate indices for the slice shape, all startes from 0
    indices = np.indices(slice_shape)
    # Adjust each dimension's indices to start from the slice's start position
    multi_index = tuple(indices[i] + s.start for i, s in enumerate(slices))
    return multi_index


# @njit
# def seed_to_rand(seeds):
#     random_matrix = np.empty(seeds.shape, dtype=np.float32)
#     for idx in np.ndindex(seeds.shape):
#         np.random.seed(seeds[idx]) 
#         random_matrix[idx] = np.random.rand()
#     return random_matrix


def rand(
        name: str, 
        shape: tuple, 
        slices: Optional[Tuple[slice, ...]] = None,
        backend: str = "cxx",
    ) -> torch.Tensor:
    """Generate a random tensor based on a unique name and shape.

    Args:
        name (str): Unique identifier for the tensor
        shape (tuple): Shape of the full tensor
        slices (Tuple[slice]): Tuple of slices to return the random value.
                               If set to None, the full tensor will be returned.
        backend (str): Backend for random tensor generation.  C++ is fast, numpy is slow.
    """

    # Giving no slices returns the full tensor
    if slices is None:
        slices = tuple(slice(0, d) for d in shape)
    slice_shape = tuple((s.stop - s.start) for s in slices)

    seed = xxhash.xxh64(name).intdigest()
    seeds = np.zeros(slice_shape, dtype=np.uint64)

    mul_index = _slice_to_index(slices)
    linear_index = np.ravel_multi_index(mul_index, shape).astype(np.uint64)
    
    # Create unique seed for each index. 
    # May overflow but we use uint64 so it is mod by 2^64
    seeds = seed + linear_index

    if backend == "numpy":
        random_matrix = seed_to_rand(seeds)
        return torch.from_numpy(random_matrix)
    elif backend == "cxx":
        return seed_to_rand_parallel(torch.from_numpy(seeds.astype(np.int64)))
    else:
        assert False, f"Backend {backend} is not supported."
