import weakref
import torch
from dataclasses import dataclass
from collections import OrderedDict
from typing import Tuple


_finalizers = {}
REG_PARAM = OrderedDict()  # {canonical_name: TensorInfo, id(Tensor): TensorInfo}


@dataclass
class TensorInfo:
    tensor_id: int  # id(tensor), avoid preventing tensor from being released
    canonical_name: str
    full_shape: Tuple[int, ...]
    shard_slice: Tuple[slice, ...]
    dtype: torch.dtype
    # device: torch.device
    local_name: str = ""  # don't know what to do with it...


def _cleanup_callback(tensor_id):
    """
    Remove the tensor and finalizer from the registry to release memory.
    """
    canonical_name = REG_PARAM[tensor_id].canonical_name
    del REG_PARAM[canonical_name]
    del REG_PARAM[tensor_id]
    del _finalizers[tensor_id]


def register_tensor( 
        tensor: torch.Tensor,
        tensor_info: TensorInfo, 
        destination: OrderedDict
    ) -> None:
    """
    Register a tensor with a name.
    Need this functionality because cannot add a name field to the tensor 
    (will lose it in tensor.to())
    """
    tid = tensor_info.tensor_id
    destination[tensor_info.canonical_name] = tensor_info
    destination[tid] = tensor_info
    _finalizers[tid] = weakref.finalize(tensor, _cleanup_callback, tid)
    