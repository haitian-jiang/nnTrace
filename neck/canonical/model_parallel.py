import torch
from torch.distributed.tensor import DTensor
from typing import Optional
from neck import CONFIG
from neck.structs import TensorAnnotation, ShardNDSlices, TensorInfo
from .megatron import (
    to_canonical_param_mod_megatron, 
    infer_slice_megatron,
    trim_word_embeddings_megatron,
)


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
    elif CONFIG.parallel.framework == "vLLM":
        return local_name  # TODO: single card for now.
    elif CONFIG.parallel.framework == "transformers":
        return local_name  # No PP for now.
    elif CONFIG.parallel.framework == "TorchTitan":
        return local_name  # TorchTitan uses ModuleDict, no need to convert.
    else:
        raise NotImplementedError(
            f"Parallel framework {CONFIG.parallel.framework} is not supported."
        )


def infer_slice(
    local_tensor: torch.Tensor | DTensor,
    annotation: Optional[TensorAnnotation] = None,
) -> ShardNDSlices:
    """
    Infer the full shape and slice for a tensor given its runtime shape and 
    annotation information.  If the annotation is None, then the runtime shape
    is the full shape and the slice.
    """
    if CONFIG.parallel.framework == "Megatron-LM":
        return infer_slice_megatron(local_tensor.shape, annotation)
    elif CONFIG.parallel.framework == "vLLM":
        return ShardNDSlices(local_tensor.shape)  # TODO: single card for now.
    elif CONFIG.parallel.framework == "transformers":
        from .transformers import infer_slice_transformers
        return infer_slice_transformers(local_tensor.shape, annotation)
    elif CONFIG.parallel.framework == "TorchTitan":
        from .torchtitan import infer_slice_torchtitan
        return infer_slice_torchtitan(local_tensor, annotation)
    else:
        raise NotImplementedError(
            f"Parallel framework {CONFIG.parallel.framework} is not supported."
        )


def trim_word_embeddings(
    info_dict: dict[str, TensorInfo]
) -> dict[str, TensorInfo]:
    """
    Trim the word embeddings tensor to the vocabulary size.
    """
    if CONFIG.parallel.framework == "Megatron-LM":
        return trim_word_embeddings_megatron(info_dict)
    elif CONFIG.parallel.framework in ["vLLM", "transformers"]:
        return info_dict  # No trimming vLLM.
    else:
        return info_dict