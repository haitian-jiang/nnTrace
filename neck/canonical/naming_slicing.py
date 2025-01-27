import torch
from neck import CONFIG
from typing import Tuple, Optional
from .megatron import to_canonical_name_param_megatron, get_tp_slice_megatron


def to_canonical_name_param(name: str, vp_rank: Optional[int] = None) -> str:
    """Convert a weight name to a canonical name."""
    if CONFIG.parallel.framework == "Megatron-LM":
        return to_canonical_name_param_megatron(name, vp_rank)
    else:
        raise NotImplementedError(
            f"Parallel framework {CONFIG.parallel.framework} is not supported."
        )


def get_tp_slice(
        name: str, 
        parameter: torch.Tensor = None
    ) -> Tuple[Tuple[int, ...], Tuple[slice, ...]]:
    if CONFIG.parallel.framework == "Megatron-LM":
        return get_tp_slice_megatron(name, parameter)
    else:
        raise NotImplementedError(
            f"Parallel framework {CONFIG.parallel.framework} is not supported."
        )
