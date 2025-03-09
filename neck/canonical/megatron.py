import torch
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



def get_megatron_lm_parallel_rank() -> OmegaConf:

    from megatron.core import parallel_state
    assert parallel_state.is_initialized(), \
        "Megatron-LM parallel state is not initialized."
    
    rank = CONFIG.parallel.rank
    size = CONFIG.parallel.size
    
    rank.world_rank = torch.distributed.get_rank()
    rank.dp_rank = parallel_state.get_data_parallel_rank()
    rank.cp_rank = parallel_state.get_context_parallel_rank()
    rank.tp_rank = parallel_state.get_tensor_model_parallel_rank()
    rank.pp_rank = parallel_state.get_pipeline_model_parallel_rank()

    size.world_size = torch.distributed.get_world_size()
    size.dp_size = parallel_state.get_data_parallel_world_size()
    size.cp_size = parallel_state.get_context_parallel_world_size()
    size.tp_size = parallel_state.get_tensor_model_parallel_world_size()
    size.pp_size = parallel_state.get_pipeline_model_parallel_world_size()
    size.vp_size = parallel_state.get_virtual_pipeline_model_parallel_world_size()

    return CONFIG.parallel


def get_megatron_parallel_options() -> None:
    """
    Get the parallel options for Megatron-LM and store in CONFIG.parallel.
    For example: tp, sp, dp, cp, pp, vp.
    Need to run after get_megatron_lm_parallel_rank.
    """
    CONFIG.parallel.types = []
    sizes = CONFIG.parallel.size
    for xp in ["tp", "dp", "pp", "vp", "cp"]:
        if sizes[f"{xp}_size"] and sizes[f"{xp}_size"] > 1:
            CONFIG.parallel.types.append(xp)
    sp = CONFIG.parallel.get("sequence_parallel", None)
    if sp:
        CONFIG.parallel.types.append("sp")


def to_canonical_param_mod_megatron(
        local_name: str, 
        vp_rank: int,
    ) -> str:
    """
    Convert a parameter name to canonical form, 
    with the info of pipeline parallel rank.
    See megatron/core/transformer/transformer_block.py/@get_num_layers_to_build
    Assume the all numbers of layers can be divided up.  
    This is guaranteed in megatron/training/arguments.py/@validate_args

                    layer/global_vp_id
    pp_rk\vp_rk    0           1           2
    stg0      (0,1)/0    (08,09)/4    (16,17)/8
    stg1      (2,3)/1    (10,11)/5    (18,19)/9
    stg2      (4,5)/2    (12;13)/6    (20,21)/10
    stg3      (6,7)/3    (14,15)/7    (22,23)/11
    """
    assert vp_rank >= 0, "vp_rank wrong."
    
    name_decomp = local_name.split(".")
    layer_id_index, local_layer_id = None, None
    for i, part in enumerate(name_decomp):
        if part.isdigit():
            layer_id_index = i
            local_layer_id = int(part)
            break
    if layer_id_index is None:
        # position_embeddings, word_embeddings, output_layer, etc. 
        if local_name.endswith("output_layer.weight") and \
            CONFIG.architecture.share_embeddings_and_output_weights:
            # *.output_layer.weight -> *embedding.word_embeddings.weight
            # output_layer and word_embeddings will not coexist
            # in the same process if share_embeddings_and_output_weights,
            # so there will not be key collision.
            name_decomp[-2] = "embedding.word_embeddings"
            return ".".join(name_decomp)
        return local_name 
    
    vp_size = CONFIG.parallel.size.vp_size
    if vp_size is None:
        vp_size = 1
    
    pp_size = CONFIG.parallel.size.pp_size
    pp_rank = CONFIG.parallel.rank.pp_rank

    # copied from `megatron/core/transformer/transformer_block.py/@get_num_layers_to_build`
    num_layers_per_pipeline_rank = CONFIG.architecture.num_layers // pp_size
    num_layers_per_virtual_rank = num_layers_per_pipeline_rank // vp_size

    global_vp_id = vp_rank * pp_size + pp_rank
    global_layer_id = global_vp_id * num_layers_per_virtual_rank + local_layer_id

    name_decomp[layer_id_index] = str(global_layer_id)
    return ".".join(name_decomp)


    # # checking, if full_shape and slices are proveded in annotation
    # if annotation.full is not None or annotation.slice is not None:
    #     hidden_size = CONFIG.architecture.hidden_size
    #     seq_length = CONFIG.architecture.seq_length
    #     _allowed_locals = {
    #         "tp_size": tp_size,
    #         "hidden_size": hidden_size,
    #         "seq_length": seq_length
    #     }
    
    # if annotation.full is not None:
    #     annotation_full_shape = eval(annotation.full, {}, _allowed_locals)
    #     annotation_full_shape = torch.Size(annotation_full_shape)
    #     assert full_shape == annotation_full_shape, \
    #         f"Full shape mismatch: {full_shape} vs {annotation_full_shape}"
        
    # if annotation.slice is not None:
    #     annotation_slice_shape = eval(annotation.slice, {}, _allowed_locals)
    #     assert runtime_shape == annotation_slice_shape, \
    #         f"Slice mismatch: {ndslice} vs {annotation_slice_shape}"


def infer_slice_megatron(
        local_shape: torch.Size,
        annotation: Optional[TensorAnnotation] = None,
    ) -> ShardNDSlices:
    """
    The mapping from TP / SP ranks to the slice of tensor in Megatron.
    """

    if annotation is None:  # no sharding
        return ShardNDSlices(local_shape)
    
    tp_dim = annotation.tp_dim  # None if not sharded, otherwise int
    sp_dim = annotation.sp_dim  # None if not sharded, otherwise 0
    cp_dim = annotation.cp_dim  # None if not sharded, otherwise int
    assert sp_dim is None or sp_dim == 0, "SP should shard the first dim."
    assert not (tp_dim is not None and sp_dim is not None), \
        "TP and SP should not shard the same tensor."
    # TODO: I want to assert if CP is on, sp_dim enabled => cp_dim enabled
    tp_size = CONFIG.parallel.size.tp_size
    tp_rank = CONFIG.parallel.rank.tp_rank
    cp_size = CONFIG.parallel.size.cp_size
    cp_rank = CONFIG.parallel.rank.cp_rank

    # calculate the full shape
    full_shape = list(local_shape)
    if tp_dim is not None:
        full_shape[tp_dim] *= tp_size
    if sp_dim is not None:
        full_shape[sp_dim] *= tp_size
    if cp_dim is not None:
        full_shape[cp_dim] *= cp_size
    full_shape = torch.Size(full_shape)

    # calculate the slice, fast path of only TP and optionally SP
    if cp_dim is None:
        global_ = [NeckSlice(0, dim) for dim in local_shape]
        if tp_dim is not None:
            global_[tp_dim] = NeckSlice(local_shape[tp_dim] * tp_rank, 
                                        local_shape[tp_dim] * (tp_rank + 1))
        if sp_dim is not None:
            global_[sp_dim] = NeckSlice(local_shape[sp_dim] * tp_rank, 
                                        local_shape[sp_dim] * (tp_rank + 1))
        slice_map = {NDSlice(local_shape): NDSlice(global_)}
        return ShardNDSlices(full_shape, local_shape, slice_map)

    # CP is enabled, has two blocks in slice_map if sans SP, one if with SP
    # stripped attention for decoder CP. TODO: see if general enough.
    global_h = [NeckSlice(0, dim) for dim in local_shape]
    global_t = [NeckSlice(0, dim) for dim in local_shape]
    seq_len = full_shape[cp_dim]
    cp_seq_blk = seq_len // (cp_size * 2)  # 2 blocks
    
    # with CP, sans TP, sans SP
    global_h[cp_dim] = NeckSlice(cp_seq_blk * cp_rank,
                                 cp_seq_blk * (cp_rank + 1))
    global_t[cp_dim] = NeckSlice(seq_len - cp_seq_blk * (cp_rank + 1),
                                 seq_len - cp_seq_blk * cp_rank)
    
    if sp_dim is None and tp_dim is not None:
    # no SP, optionally TP.  If no TP, don't do anything; if TP, slice TP.
        # with CP, with TP, sans SP
        global_h[tp_dim] = NeckSlice(local_shape[tp_dim] * tp_rank,
                                     local_shape[tp_dim] * (tp_rank + 1))
        global_t[tp_dim] = NeckSlice(local_shape[tp_dim] * tp_rank,
                                     local_shape[tp_dim] * (tp_rank + 1))
    if sp_dim is None:
        assert local_shape[cp_dim] == cp_seq_blk * 2, \
            f"Wrong shape for CP, sans SP."
        local_h = [NeckSlice(0, dim) for dim in local_shape]
        local_t = [NeckSlice(0, dim) for dim in local_shape]
        local_h[cp_dim] = NeckSlice(0, cp_seq_blk)
        local_t[cp_dim] = NeckSlice(cp_seq_blk, cp_seq_blk * 2)
        slice_map = {NDSlice(local_h): NDSlice(global_h),
                     NDSlice(local_t): NDSlice(global_t)}
        return ShardNDSlices(full_shape, local_shape, slice_map)

    # with CP, with SP, optionally TP
    assert sp_dim == cp_dim, "CP and SP should shard the same dim."
    seq_dim = cp_dim
    assert local_shape[seq_dim] * tp_size == cp_seq_blk * 2, \
        f"Wrong shape for CP, with SP."
    cp_blk_id = tp_rank // (tp_size // 2)  # 0: head block, 1: tail block (stripped attn)
    sp_id_within_cp_blk = tp_rank % (tp_size // 2)
    global_ = global_h if cp_blk_id == 0 else global_t  # to be modified
    cp_seq_start = global_[seq_dim].start  # offset for the same cp block (h/t)
    local_seq_len = local_shape[seq_dim]  # SP > 2, so only 1 slice in slice_map
    local_seq_start = cp_seq_start + local_seq_len * sp_id_within_cp_blk
    local_seq_stop = local_seq_start + local_seq_len
    global_[seq_dim] = NeckSlice(local_seq_start, local_seq_stop)
    slice_map = {NDSlice(local_shape): NDSlice(global_)}
    return ShardNDSlices(full_shape, local_shape, slice_map)


def _rescale_grad_for_pp_microbatch_hook(
        grad_info: TensorInfo, 
        t_type: TType,
    ) -> TensorInfo:
    """
    Log hook. Rescale the gradient tensor. In Megatron, during PP, the 
    output_tensor is divided by the number of microbatches, so the gradients
    are shrunk. This hook rescales the gradient tensor to compensate for that.

    Args:
        grad: The gradient tensor being shrunk by Megatron.

    Returns:
        The original gradient tensor.
    """
    if t_type not in [TType.g_activ, TType.g_param]:
        return grad_info
    # only for grad obtained by torch hooks
    try:
        from megatron.core.num_microbatches_calculator import get_num_microbatches
        num_microbatches = get_num_microbatches()
    except ImportError:
        num_microbatches = CONFIG.parallel.num_microbatches

    if num_microbatches > 1:
        grad_info.tensor *= num_microbatches
    return grad_info


def _rescale_grad_for_loss_cp_rescale_hook(
        grad_info: TensorInfo,
        t_type: TType,
    ) -> TensorInfo:
    """
    Log hook. Rescale the gradient tensor. In Megatron, if CP is enabled, the
    loss tensor is multipled by the number of CP ranks, so the gradients are
    inflated. This hook rescales the gradient tensor to compensate for that.
    See Megatron's pretrain_gpt.py/@loss_fun.
    """
    if t_type not in [TType.g_activ, TType.g_param]:
        return grad_info
    # only for grad obtained by torch hooks
    cp_size = CONFIG.parallel.size.cp_size
    if cp_size > 1:
        grad_info.tensor /= cp_size
    return grad_info


_need_finalize_suffix = [
    "layernorm", "layer_norm", 
    "self_attention.linear_proj.bias", 
    "mlp.linear_fc2.bias", 
    "embedding.word_embeddings.weight",
    "embedding.position_embeddings.weight",
]
def _finalize_model_grads_megatron(
        grad_info: TensorInfo, 
        t_type: TType,
    ) -> TensorInfo:
    """
    Some gradients are synced before grad update in `finalize_model_grads`.
    For example, the layernorm gradients and embedding+lm_head grad.
    These gradients has full shape, and has different value on different ranks.
    Need to leave information for the merger later to add them together.
    """
    if t_type not in [TType.g_param]:
        return grad_info
    c_name = grad_info.canonical_name
    if any(suffix in c_name for suffix in _need_finalize_suffix):
        grad_info.extra["need_finalize_grad"] = True
    return grad_info


log_hook_megatron = [_finalize_model_grads_megatron]  
if True:  # TODO: change to NO calculate-per-tok-loss
    # NOTE: mostly cp_size will be a power of 2, so the division is exact.
    log_hook_megatron.append(_rescale_grad_for_pp_microbatch_hook)
    log_hook_megatron.append(_rescale_grad_for_loss_cp_rescale_hook)


__all__ = [
    "get_megatron_lm_parallel_rank",
    "to_canonical_param_mod_megatron",
    "infer_slice_megatron",
    "log_hook_megatron",
]
