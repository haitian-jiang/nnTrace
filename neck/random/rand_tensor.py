import numpy as np
import torch
import xxhash
from typing import Optional
from neck.structs import ShardNDSlices
from ..cpp_ext import seed_to_rand_parallel


def rand(
    name: str, 
    shard_slice: ShardNDSlices,
    device="cpu",
    dtype: torch.dtype = torch.float32,
    backend: str = "pytorch",
) -> torch.Tensor:
    """Generate a random tensor based on a unique name and shape.

    Args:
        name (str): Unique identifier for the tensor
        shape (tuple): Shape of the full tensor
        slices (Tuple[slice]): Tuple of slices to return the random value.
                               If set to None, the full tensor will be returned.
        backend (str): Backend for random tensor generation.  C++ is fast.
    """

    seed = xxhash.xxh64(name).intdigest()
    if backend == "pytorch":
        g = torch.Generator(device=device)
        g.manual_seed(seed)
        result = torch.zeros(shard_slice.local_shape, device=device)
        virtual_full_matrix = torch.rand(shard_slice.full_shape, generator=g, 
                                         device=device, dtype=dtype)
        for local_, global_ in shard_slice.slice_map.items():
            result[local_.pyslice] = virtual_full_matrix[global_.pyslice]
        return result.contiguous()
    
    linear_index_aggr = np.zeros(shard_slice.local_shape, dtype=np.uint64)
    seeds = np.zeros(shard_slice.local_shape, dtype=np.uint64)
    for local_, global_ in shard_slice.slice_map.items():
        # local_ and global_ have the same shape
        mul_index = global_.multi_index
        linear_index = np.ravel_multi_index(
            mul_index, 
            shard_slice.full_shape
        ).astype(np.uint64)
        linear_index_aggr[local_.pyslice] = linear_index
    
    # Create unique seed for each index. 
    # May overflow but we use uint64 so it is mod by 2^64
    seeds = seed + linear_index_aggr
    seeds = torch.from_numpy(seeds.astype(np.int64)).contiguous()

    if backend == "cxx":
        return seed_to_rand_parallel(seeds).to(dtype).to(device)
    else:
        assert False, f"Backend {backend} is not supported."


def randn(
    name: str,
    shard_slice: ShardNDSlices,
    device="cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Generate a random tensor based on a unique name and shape."""
    seed = xxhash.xxh64(name).intdigest()
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    result = torch.zeros(shard_slice.local_shape, device=device)
    virtual_full_matrix = torch.randn(shard_slice.full_shape, generator=g, 
                                      device=device, dtype=dtype)
    for local_, global_ in shard_slice.slice_map.items():
        result[local_.pyslice] = virtual_full_matrix[global_.pyslice]
    return result.contiguous()


def rand_mask(
    name: str,
    shard_slice: ShardNDSlices,
    p: float = 1.0,
    device="cpu",
) -> torch.Tensor:
    """
    Args:
        p: Probability of being reserved.

    Returns:
        torch.Tensor: A boolean tensor with True at the reserved positions.
    """
    seed = xxhash.xxh64(name).intdigest()
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    result = torch.zeros(shard_slice.local_shape, 
                         dtype=torch.bool, device=device)
    virtual_full_matrix = torch.rand(shard_slice.full_shape, 
                                     generator=g, device=device) < p
    for local_, global_ in shard_slice.slice_map.items():
        result[local_.pyslice] = virtual_full_matrix[global_.pyslice]
    return result.contiguous()


__all__ = ["rand", "randn", "rand_mask"]
