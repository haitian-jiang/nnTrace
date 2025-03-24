import numpy as np
import torch
import xxhash
from typing import Optional
from neck.structs import ShardNDSlices
from .cpp_ext import (
    seed_to_rand_parallel, 
    rewrite_low_bits_bf16 as _rewrite_low_bits_bf16,
)


def _seed_to_seeds(seed: int, shard_slice: ShardNDSlices) -> torch.Tensor:
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
    return seeds


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
    elif backend == "cxx":
        seeds = _seed_to_seeds(seed, shard_slice)
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


def rewrite_low_bits(
    name: str,
    shard_slice: ShardNDSlices,
    activation: torch.Tensor,
    nbits: Optional[int] = 2,
):
    assert activation.dtype in [torch.float16, torch.bfloat16], \
        "Activation tensor must be float16 or bfloat16 for low bits rewrite."
    seed = xxhash.xxh64(name).intdigest()
    seeds = _seed_to_seeds(seed, shard_slice).to(activation.device)
    
    if activation.dtype == torch.bfloat16:
        _rewrite_low_bits_bf16(activation, seeds, nbits)
    elif activation.dtype == torch.float16:
        raise NotImplementedError("Low bits rewrite for float16 is not implemented.")
    else:
        raise ValueError("Activation tensor must be float16 or bfloat16.")

__all__ = [
    "rand", 
    "randn", 
    "rand_mask",
    "rewrite_low_bits",
]
