import torch
from dataclasses import dataclass
from typing import Tuple, List
from .slices import ShardSlice
from loguru import logger as clog


@dataclass
class TensorInfo:
    tensor: torch.Tensor  # CPU tensor, not the original tensor
    canonical_name: str
    # full_shape: Tuple[int, ...]
    # shard_slice: Tuple[slice, ...]
    shape: ShardSlice
    module_class: str = ""  # empty for weight tensors
    is_grad: bool = False  # g_activ or g_param

    @staticmethod
    def _multiple_grads_each_shard(t_infos: List["TensorInfo"]) -> "TensorInfo":
        """
        The input tensor info lists contains the same amount (>1) of grads for
        each shard.  Check and add them into one tensor info for each shard.
        NOTE: This should NOT happen if the framework is compatible with pytorch.
        It happened in Megatron-LM when optimization turned on, and the gradeint
        obtained from pytorch hook is junck, or one grad is 0 for word embedding.
        Have this additional checking just in case, making the code no crash.
        """
        # ShardSlice as key, store a list of TensorInfo with same ShardSlice.
        info_dict = {}
        for info in t_infos:
            shard = info.shape
            zero = torch.zeros_like(info.tensor)
            if shard not in info_dict:
                info_dict[shard] = info
            elif torch.allclose(info_dict[shard].tensor, info.tensor):
                clog.info(f"Multiple tensors for the same shard. 
                          Same value, abandon one: {info.canonical_name}")
            elif torch.allclose(info.tensor, zero):
                clog.info(f"Multiple tensors for the same shard. 
                          One is zero, abandon it: {info.canonical_name}")
            elif torch.allclose(info_dict[shard].tensor, zero):
                clog.info(f"Multiple tensors for the same shard. 
                          One is zero, abandon it: {info.canonical_name}")
                info_dict[shard] = info
            else: # Same shard, different grads, shouldn't happen. But add them.
                print(info_dict[shard].tensor.sum())
                print(info.tensor.sum())
                clog.warning(f"Multiple tensors for the same shard.  
                             Different value, add them: {info.canonical_name}")
                info_dict[shard].tensor += info.tensor
        return list(info_dict.values())

    @staticmethod
    def merge_tensor_info(t_infos: List["TensorInfo"]) -> "TensorInfo":
        """
        Merge the list of TensorInfo from shards into one TensorInfo for the
        full tensor.
        """
        # checking
        if not all(s.canonical_name == t_infos[0].canonical_name for s in t_infos):
            raise ValueError("Cannot merge tensors with different names.")
        if not all(s.is_grad == t_infos[0].is_grad for s in t_infos):
            raise ValueError("Cannot merge tensors with different grad types.")
        if not all(s.module_class == t_infos[0].module_class for s in t_infos):
            raise ValueError("Cannot merge tensors from different modules.")
        slices = [t.shape for t in t_infos]
        full_shape = slices[0].full_shape
        if not all(s.full_shape == full_shape for s in slices):
            raise ValueError("Cannot merge tensors with different shapes.")
        mask = ShardSlice.get_completeness(slices)
        if torch.any(mask == 0):
            raise ValueError("Cannot merge incomplete tensors.")
        elif not torch.all(mask == mask.flatten()[0]):
            raise ValueError("Cannot merge tensors with different completeness.")
        elif mask.flatten()[0] > 1:
            t_infos = TensorInfo._multiple_grads_each_shard(t_infos)

        # merge
        result = torch.zeros(full_shape, dtype=t_infos[0].tensor.dtype)
        for info in t_infos:
            result[info.shape.shard_slice] = info.tensor
        return TensorInfo(
            tensor=result,
            canonical_name=t_infos[0].canonical_name,
            shape=ShardSlice(None, full_shape),  # already checked completeness
            module_class=t_infos[0].module_class,
            is_grad=t_infos[0].is_grad,
        )

T_TYPE = ("activ", "g_activ", "g_param", "param", "g_param_accum")


__all__ = ["TensorInfo", "T_TYPE"]
