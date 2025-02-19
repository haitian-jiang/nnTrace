import torch
from dataclasses import dataclass
from typing import Tuple
from .slices import ShardSlice


@dataclass
class TensorInfo:
    tensor: torch.Tensor  # CPU tensor, not the original tensor
    canonical_name: str
    # full_shape: Tuple[int, ...]
    # shard_slice: Tuple[slice, ...]
    shape: ShardSlice
    module_class: str = ""  # empty for weight tensors
    is_grad: bool = False  # g_activ or g_param


T_TYPE = ("activ", "g_activ", "g_param", "param")


__all__ = ["TensorInfo", "T_TYPE"]
