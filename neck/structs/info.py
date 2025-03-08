import torch
from collections import defaultdict
from dataclasses import dataclass, field
from .slices import ShardNDSlices
from loguru import logger as clog
from enum import Enum


@dataclass
class TensorInfo:
    """
    Data structure to store the tensor as logs.

    Args:
        tensor: CPU tensor, not the original tensor
        canonical_name: -
        shape: -
        module_class: where the activ comes from, empty for weight tensors
        is_grad: True for "g_activ" or "g_param" or "g_param_main"
        extra: extra information, e.g., need finalize grad, if input rewriten,
               Need to be consistent with the inspector's merger.
               Allowed fields: 
               need_finalize_grad (bool) for grad to conduct final allreduce.
    """
    tensor: torch.Tensor
    canonical_name: str
    shard_slice: ShardNDSlices
    module_class: str = ""
    is_grad: bool = False
    extra: dict = field(default_factory=dict)

    def info_equal(self, other: "TensorInfo") -> bool:
        """
        Check if the tensor info is equal to another tensor info.  
        The tensor is not checked.
        """
        return (
            self.canonical_name == other.canonical_name and
            self.shard_slice == other.shard_slice and
            self.module_class == other.module_class and
            self.is_grad == other.is_grad and
            self.extra == other.extra
        )

    def __str__(self):
        return f"\n{self.tensor}\n{self.canonical_name}\n{self.shard_slice}\n" \
               f"{self.module_class}\n{self.is_grad}\n{self.extra}\n"

    def __repr__(self):
        return self.__str__()

    @staticmethod
    def _multiple_grads_each_shard(t_infos: list["TensorInfo"]) -> "TensorInfo":
        """
        Helper function for merge_tensor_info.
        The input tensor info lists contains the same amount (>1) of grads for
        each shard.  Check and add them into one tensor info for each shard.
        NOTE: This should NOT happen if the framework is compatible with pytorch.
        It happened in Megatron-LM when optimization turned on, and the gradeint
        obtained from pytorch hook is junck, or one grad is 0 for word embedding.
        Have this additional checking just in case, making the code no crash.
        """
        # ShardSlice as key, store a list of TensorInfo with same ShardSlice.
        multi_info_dict: dict[ShardNDSlices, list[TensorInfo]] = defaultdict(list)
        for info in t_infos:
            shard = info.shard_slice
            multi_info_dict[shard].append(info)
        
        # Merge the multiple grads for each shard, store the single TensorInfo.
        # If all grads are the same, keep one. 
        # If one is zero, abandon it.
        # If different and need_finalize_grad, add them. 
        # Otherwise, log error.
        info_dict = {}
        for shard, infos in multi_info_dict.items():
            assert len(infos) > 1, "Should have multiple grads for each shard."
            # all are the same, keep one.
            if all(
                torch.allclose(infos[0].tensor, info.tensor) 
                for info in infos[1:]
            ):
                info_dict[shard] = infos[0]
                continue

            # filter out the zero tensor
            zero = torch.zeros_like(infos[0].tensor)
            non_zero_infos = [
                info for info in infos 
                if not torch.allclose(info.tensor, zero)
            ]

            # Only one non-zero tensor, use it.
            if len(non_zero_infos) == 1:
                info_dict[shard] = non_zero_infos[0]
                continue

            # More than one non-zero tensor, add them if allowed.
            if all(
                info.extra.get("need_finalize_grad", False)
                for info in non_zero_infos
            ):
                info_dict[shard] = TensorInfo.add_tensor_info(non_zero_infos)
            else:  # conflicting tensors, log error
                clog.error("Conflicting tensors for the same shard: "
                           f"{infos[0].canonical_name} {shard}")
                # TODO: put a placeholder, can change to all zero or all nan later.
                info_dict[shard] = non_zero_infos[0]  
        # end for
        return list(info_dict.values())

    @staticmethod
    def merge_tensor_info(t_infos: list["TensorInfo"]) -> "TensorInfo":
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
        slices = [t.shard_slice for t in t_infos]
        full_shape = slices[0].full_shape
        if not all(s.full_shape == full_shape for s in slices):
            raise ValueError("Cannot merge tensors with different shapes.")
        mask = ShardNDSlices.get_completeness(slices)
        if torch.any(mask == 0):
            raise ValueError("Cannot merge incomplete tensors.")
        elif not torch.all(mask == mask.flatten()[0]):
            raise ValueError("Cannot merge tensors with different completeness.")
        elif mask.flatten()[0] > 1:
            t_infos = TensorInfo._multiple_grads_each_shard(t_infos)

        # merge
        result = torch.zeros(full_shape, dtype=t_infos[0].tensor.dtype)
        for info in t_infos:
            for local_, global_ in info.shard_slice.slice_map.items():
                result[global_.pyslice] = info.tensor[local_.pyslice]
            # result[info.shard_slice.shard_slice] = info.tensor
        return TensorInfo(
            tensor=result,
            canonical_name=t_infos[0].canonical_name,
            shard_slice=ShardNDSlices(full_shape),  # already checked completeness
            module_class=t_infos[0].module_class,
            is_grad=t_infos[0].is_grad,
        )
    
    @staticmethod
    def add_tensor_info(t_infos: list["TensorInfo"]) -> "TensorInfo":
        """
        Add the list of TensorInfo into one TensorInfo, need to have the same
        ShardSlice.
        Use tree-based reduction to minimize numerical error.
        """
        assert all(t.info_equal(t_infos[0]) for t in t_infos), \
            "Must finalize(add) tensors with same information."
        tmp_tensors = [t.tensor.to(torch.float64) for t in t_infos]
        # add the neighboring two tensors, do it recursively until only one left
        while len(tmp_tensors) > 1:
            new_list = []
            for i in range(0, len(tmp_tensors), 2):
                if i == len(tmp_tensors) - 1:
                    new_list.append(tmp_tensors[i])
                else:
                    new_list.append(tmp_tensors[i] + tmp_tensors[i + 1])
            tmp_tensors = new_list
        return TensorInfo(
            tensor=tmp_tensors[0],  # do not convert back, keep high precision
            canonical_name=t_infos[0].canonical_name,
            shard_slice=t_infos[0].shard_slice,
            module_class=t_infos[0].module_class,
            is_grad=t_infos[0].is_grad,
        )


class TType(str, Enum):
    """
    Enum for tensor types.  For compatability with python<=3.10, because StrEnum
    is available for python>=3.11.

    Attributes:
        activ: Activation for each microbatch during forward.
        g_activ: Gradients of activations for each microbatch during backward.
        g_param: Gradients of parameters for each microbatch during backward.
        param: Parameters upon reinitialization or after optimizer step.
        g_param_main: Main gradients of parameters before optimizer step,
                      accumulated for multiple microbatches (PP*DP).
    """
    activ = "activ"
    g_activ = "g_activ"
    g_param = "g_param"
    param = "param"
    g_param_main = "g_param_main"

    def __str__(self):
        return self.value
    
    def __repr__(self):
        return self.value


__all__ = ["TensorInfo", "TType"]
