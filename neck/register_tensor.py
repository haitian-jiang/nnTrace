import torch
from dataclasses import dataclass
from typing import Tuple


REGISTERED_TENSOR = {}


@dataclass
class TensorInfo:
    tensor: torch.Tensor
    name: str
    full_shape: Tuple[int, ...]
    shard_slice: Tuple[slice, ...]
    dtype: torch.dtype
    device: torch.device
    tp_rank: int = 0
    tp_size: int = 1


def register_tensor(name: str, tensor: torch.Tensor):
    """Register a tensor with a name."""
    if name in REGISTERED_TENSOR:
        raise ValueError(f"Tensor with name {name} already registered.")
    REGISTERED_TENSOR[name] = TensorInfo(
        tensor=tensor,
        name=name,
        full_shape=tensor.shape,
        shard_slice=None,  # TODO implement sharding
        dtype=tensor.dtype,
        device=tensor.device,
    )
    return tensor

def register_model(model: torch.nn.Module, prefix: str = ""):
    """Register all tensors in a model."""
    if prefix:
        prefix = f"{prefix}."
    for name, parameter in model.named_parameters():
        register_tensor(f"{prefix}{name}", parameter)