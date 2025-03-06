import torch
from typing import Optional
from neck import CONFIG
from neck.structs import TensorAnnotation
from .megatron import to_canonical_param_mod_megatron, infer_tp_slice_megatron


def to_canonical_param_mod(
        local_name: str, 
        vp_rank: int,
    ) -> str:
    """
    Convert a weight name or a module name to a canonical name.
    Example of a weight name: 
        module.module.decoder.layers.0.self_attention.linear_qkv.weight
    Example of a module name:
        module.module.decoder.layers.0.self_attention
    """
    if CONFIG.parallel.framework == "Megatron-LM":
        return to_canonical_param_mod_megatron(local_name, vp_rank)
    else:
        raise NotImplementedError(
            f"Parallel framework {CONFIG.parallel.framework} is not supported."
        )


def infer_tp_slice(
        runtime_shape: torch.Size,
        annotation: Optional[TensorAnnotation] = None,
    ) -> tuple[tuple[int, ...], tuple[slice, ...]]:
    """
    Infer the full shape and slice for a tensor given its runtime shape and 
    annotation information.  If the annotation is None, then the runtime shape
    is the full shape and the slice.
    """
    if CONFIG.parallel.framework == "Megatron-LM":
        return infer_tp_slice_megatron(runtime_shape, annotation)
    else:
        raise NotImplementedError(
            f"Parallel framework {CONFIG.parallel.framework} is not supported."
        )
