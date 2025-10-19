import torch
from typing import Optional
from collections import defaultdict
from torch.distributed import DeviceMesh
from omegaconf import OmegaConf
from neck import CONFIG
from neck.structs import (
    TensorAnnotation,
    TensorInfo, 
    TType, 
    NeckSlice,
    NDSlice,
    ShardNDSlices, 
    to_general_name
)


def get_transformers_parallel_rank(device_mesh: DeviceMesh | None) -> OmegaConf:
    rank = CONFIG.parallel.rank
    size = CONFIG.parallel.size

    rank.world_rank, size.world_size = 0, 1
    rank.dp_rank, size.dp_size = 0, 1
    rank.pp_rank, size.pp_size = 0, 1
    rank.tp_rank, size.tp_size = 0, 1
    size.vp_size = None

    if device_mesh is None:
        CONFIG.parallel.types = []
        return CONFIG.parallel

    assert torch.distributed.is_initialized(), "Torch distributed is not initialized"

    # get_transformers_parallel_options
    if "types" not in CONFIG.parallel:
        CONFIG.parallel.types = list(device_mesh.mesh_dim_names)  # "tp", ...

    rank.world_rank = torch.distributed.get_rank()
    size.world_size = torch.distributed.get_world_size()

    for dim, dim_name in enumerate(device_mesh.mesh_dim_names):
        rank[f"{dim_name}_rank"] = device_mesh.get_local_rank(dim)
        size[f"{dim_name}_size"] = device_mesh.shape[dim]

    return CONFIG.parallel


def infer_slice_transformers(
        local_shape: torch.Size,
        annotation: Optional[TensorAnnotation] = None,
) -> ShardNDSlices:

    if annotation is None:  # no sharding
        return ShardNDSlices(local_shape)
    # TODO 
    tp_dim = annotation.tp_dim  # currently only support TP.
    tp_size = CONFIG.parallel.size.tp_size
    tp_rank = CONFIG.parallel.rank.tp_rank

    if tp_size == 1:
        tp_dim = None
    
    full_shape = list(local_shape)
    if tp_dim is not None:
        full_shape[tp_dim] *= tp_size
    full_shape = torch.Size(full_shape)
    
    global_ = [NeckSlice(0, dim) for dim in local_shape]
    if tp_dim is not None:
        global_[tp_dim] = NeckSlice(local_shape[tp_dim] * tp_rank,
                                    local_shape[tp_dim] * (tp_rank + 1))
    slice_map = {NDSlice(local_shape): NDSlice(global_)}
    return ShardNDSlices(full_shape, local_shape, slice_map)


class reshape_q_k_v_hook:
    def __init__(self, head_dim):
        self.head_dim = head_dim

    def __call__(self, module: torch.nn.Module, args, q_k_v: torch.Tensor):
        """
        Args:
            module: q_proj or k_proj or v_proj
            q_k_v: q or k or v
            TODO: In the case num_head can not be divided by tp_size, HF will shard head_dim.
            TODO: Need to update the infer_slice.
        """
        # below copied from transformers LlamaAttention.forward
        input_shape = q_k_v.shape[:-1]  # (batch_size, sequence_length)
        hidden_shape = (*input_shape, -1, self.head_dim)
        q_k_v = q_k_v.view(hidden_shape)  # idempotent
        return q_k_v


modules_to_hook = [
    r"layers.\d.self_attn.q_proj",
    r"layers.\d.self_attn.k_proj",
    r"layers.\d.self_attn.v_proj",
]


def register_reshape_hook(module: torch.nn.Module, head_dim: int):
    hook = reshape_q_k_v_hook(head_dim)
    for name, submodule in module.named_modules():
        general_name = to_general_name(name)
        for suffix in modules_to_hook:
            if general_name.endswith(suffix):
                submodule.register_forward_hook(hook)


def post_processing_before_save_transformers(LOGS: dict[TType, dict[str, TensorInfo]]) -> None:
    activ = LOGS[TType.activ]
    temp_activ_dict: dict[str, dict[int, TensorInfo]] = defaultdict(dict)
    for cname, info in activ.items():
        # iter.6|batch.0|layers.7.self_attn.q_proj|activ.output.0
        it, batch, layer, attrib = cname.split("|")
        it = int(it.split(".")[1])  # iter.6 -> 6
        name = f"{batch}|{layer}|{attrib}"
        temp_activ_dict[name][it] = info
    
    max_steps = [max(d.keys()) for d in temp_activ_dict.values()]
    assert all(s == max_steps[0] for s in max_steps), \
        "All layers should have the same number of iterations."
    max_steps = max_steps[0]

    temp_activ_list: dict[str, list[TensorInfo]] = defaultdict(lambda: [None]*(max_steps+1))
    for name, info_dict in temp_activ_dict.items():
        for it, info in info_dict.items():
            temp_activ_list[name][it] = info

    new_activ = {}
    seq_len = 0
    for name, info_list in temp_activ_list.items():
        start_idx = 0
        prefill: torch.Tensor = info_list[0].tensor
        if seq_len == 0:
            seq_dim = 1
            seq_len = sum([info.tensor.shape[seq_dim] for info in info_list])

        concat_shape = list(prefill.shape)
        concat_shape[seq_dim] = seq_len
        concat_tensor = torch.zeros(concat_shape, dtype=prefill.dtype)
        concat_full_slice = [slice(0, d) for d in concat_shape]
        for info in info_list:
            s = info.tensor.shape[seq_dim]
            concat_full_slice[seq_dim] = slice(start_idx, start_idx + s)
            concat_tensor[tuple(concat_full_slice)] = info.tensor
            start_idx += s

        # merge the TensorInfo and ShardNDSlices
        concat_global_shape: list[int] = list(info_list[0].shard_slice.full_shape)
        concat_global_shape[seq_dim] = seq_len
        local_slice = NDSlice(concat_tensor.shape)
        global_slice: list[NeckSlice] = list(
            list(info_list[0].shard_slice.slice_map.values())[0]
        )
        global_slice[seq_dim] = NeckSlice(0, seq_len)
        concat_info: TensorInfo = TensorInfo(
            concat_tensor,  # [bs, seq, n_head, head_dim]
            info_list[0].canonical_name,
            ShardNDSlices(concat_global_shape, concat_shape, 
                          {local_slice: NDSlice(global_slice)}),
            info_list[0].module_class,
            info_list[0].is_grad,
            info_list[0].extra,
        )

        new_activ[name] = concat_info
    LOGS[TType.activ] = new_activ


__all__ = [
    "get_transformers_parallel_rank",
    "register_reshape_hook",
]
