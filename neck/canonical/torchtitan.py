import torch
import torch.distributed.tensor as dtensor
from typing import Optional
from omegaconf import OmegaConf
from neck import CONFIG
from neck.structs import (
    TensorAnnotation, 
    TensorInfo, 
    TType, 
    NeckSlice,
    NDSlice,
    ShardNDSlices,
)


_PARALLELISM_MAPPING = { # torchtitan -> neck
    "dp_replicate": "dp",
    "dp_shard": "fsdp",
    "pp": "pp",
    "tp": "tp",
    "cp": "cp",
}


def get_local_shape(tensor: torch.Tensor | dtensor.DTensor) -> torch.Size:
    if isinstance(tensor, dtensor.DTensor):
        return tensor._local_tensor.shape
    elif isinstance(tensor, torch.Tensor):
        return tensor.shape
    else:
        raise ValueError(f"Unsupported tensor type: {type(tensor)}")


def get_torchtitan_parallel_rank(world_mesh) -> OmegaConf:
    assert torch.distributed.is_initialized(), "Torch distributed is not initialized"

    rank = CONFIG.parallel.rank
    size = CONFIG.parallel.size
    rank.world_rank, size.world_size = 0, 1
    for dim_name in _PARALLELISM_MAPPING.values():
        rank[f"{dim_name}_rank"] = 0
        size[f"{dim_name}_size"] = 1
    size.vp_size = 1 << 31  # just to pass assertion check in _check_vp_rank

    if world_mesh is None or world_mesh.mesh_dim_names is None:
        CONFIG.parallel.types = []
        return CONFIG.parallel

    if "types" not in CONFIG.parallel:
        # whether vpp is used is not important, because don't need to infer name for PP.
        CONFIG.parallel.types = list(
            _PARALLELISM_MAPPING[xp] for xp in world_mesh.mesh_dim_names
        )
        # TODO: add sp

    rank.world_rank = torch.distributed.get_rank()
    size.world_size = torch.distributed.get_world_size()
    
    for dim, dim_name in enumerate(world_mesh.mesh_dim_names):
        neck_dim_name = _PARALLELISM_MAPPING[dim_name]
        rank[f"{neck_dim_name}_rank"] = world_mesh.get_local_rank(dim)
        size[f"{neck_dim_name}_size"] = world_mesh.shape[dim]

    return CONFIG.parallel


def infer_slice_torchtitan(
    local_tensor: torch.Tensor | dtensor.DTensor,
    annotation: Optional[TensorAnnotation] = None,
) -> ShardNDSlices:

    local_shape = get_local_shape(local_tensor)
    if annotation is None:
        if isinstance(local_tensor, dtensor.DTensor):
            assert all(
                p == dtensor.placement_types.Replicate 
                for p in local_tensor.placements
            ), "The tensor must be replicated if no annotation is provided."
        return ShardNDSlices(local_shape)

    tp_dim = annotation.tp_dim
    tp_size = CONFIG.parallel.size.tp_size
    tp_rank = CONFIG.parallel.rank.tp_rank

    if tp_size == 1:
        tp_dim = None
    
    full_shape = list(local_shape)
    if tp_dim is not None:
        full_shape[tp_dim] *= tp_size
    full_shape = torch.Size(full_shape)

    if isinstance(local_tensor, dtensor.DTensor):
        assert full_shape == local_tensor.shape, \
            f"Wrong shape! Inferred: {full_shape}, actual: {local_tensor.shape}"
    
    # No CP
    global_ = [NeckSlice(0, dim) for dim in local_shape]
    if tp_dim is not None:
        global_[tp_dim] = NeckSlice(local_shape[tp_dim] * tp_rank,
                                    local_shape[tp_dim] * (tp_rank + 1))
    slice_map = {NDSlice(local_shape): NDSlice(global_)}
    return ShardNDSlices(full_shape, local_shape, slice_map)
