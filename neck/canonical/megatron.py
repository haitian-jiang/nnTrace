import torch
from typing import Tuple, Optional
from omegaconf import OmegaConf
from neck import CONFIG
from neck.structs import TensorAnnotation


def get_megatron_lm_parallel_rank() -> OmegaConf:

    from megatron.core import parallel_state
    assert parallel_state.is_initialized(), \
        "Megatron-LM parallel state is not initialized."
    
    rank = CONFIG.parallel.rank
    size = CONFIG.parallel.size
    
    rank.world_rank = torch.distributed.get_rank()
    rank.dp_rank = parallel_state.get_data_parallel_rank()
    rank.tp_rank = parallel_state.get_tensor_model_parallel_rank()
    rank.pp_rank = parallel_state.get_pipeline_model_parallel_rank()

    size.world_size = torch.distributed.get_world_size()
    size.dp_size = parallel_state.get_data_parallel_world_size()
    size.tp_size = parallel_state.get_tensor_model_parallel_world_size()
    size.pp_size = parallel_state.get_pipeline_model_parallel_world_size()
    size.vp_size = parallel_state.get_virtual_pipeline_model_parallel_world_size()

    return CONFIG.parallel


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


def infer_tp_slice_megatron(
        runtime_shape: Tuple[int, ...],
        annotation: Optional[TensorAnnotation] = None,
    ) -> Tuple[Tuple[int, ...], Tuple[slice, ...]]:

    if annotation is None:
        return runtime_shape, tuple(slice(0, dim) for dim in runtime_shape)
    
    tp_dim = annotation.tp_dim  # None if not sharded, otherwise int
    tp_size = CONFIG.parallel.size.tp_size
    tp_rank = CONFIG.parallel.rank.tp_rank

    # calculate the full shape
    full_shape = list(runtime_shape)
    if tp_dim is not None:
        full_shape[tp_dim] *= tp_size
    full_shape = tuple(full_shape)

    # calculate the slice
    slices = [slice(0, dim) for dim in runtime_shape]
    if tp_dim is not None:
        slices[tp_dim] = slice(runtime_shape[tp_dim] * tp_rank, 
                            runtime_shape[tp_dim] * (tp_rank + 1))

    # checking, if full_shape and slices are proveded in annotation
    if annotation.full is not None or annotation.slice is not None:
        hidden_size = CONFIG.architecture.hidden_size
        vocab_size_with_padding = CONFIG.architecture.vocab_size_with_padding
        seq_length = CONFIG.architecture.seq_length
        _allowed_locals = {"tp_size": tp_size, "hidden_size": hidden_size,
                        "vocab_size_with_padding": vocab_size_with_padding,
                        "seq_length": seq_length}
    
    if annotation.full is not None:
        annotation_full_shape = eval(annotation.full, {}, _allowed_locals)
        assert full_shape == annotation_full_shape, \
            f"Full shape mismatch: {full_shape} vs {annotation_full_shape}"
        
    if annotation.slice is not None:
        annotation_slice_shape = eval(annotation.slice, {}, _allowed_locals)
        assert runtime_shape == annotation_slice_shape, \
            f"Slice mismatch: {slices} vs {annotation_slice_shape}"
        
    return full_shape, tuple(slices)