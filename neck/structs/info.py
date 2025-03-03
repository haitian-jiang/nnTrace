import torch
from dataclasses import dataclass, field
from .slices import ShardSlice
from loguru import logger as clog


@dataclass
class TensorInfo:
    """
    Data structure to store the tensor as logs.

    Args:
        tensor: CPU tensor, not the original tensor
        canonical_name: -
        shape: -
        module_class: where the activ comes from, empty for weight tensors
        is_grad: True for "g_activ" or "g_param" or "g_param_accum"
        extra: extra information, e.g., need finalize grad, if input rewriten,
               Need to be consistent with the inspector's merger.
               Allowed fields: 
               need_finalize_grad (bool) for grad to conduct final allreduce.
    """
    tensor: torch.Tensor
    canonical_name: str
    shape: ShardSlice
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
            self.shape == other.shape and
            self.module_class == other.module_class and
            self.is_grad == other.is_grad and
            self.extra == other.extra
        )

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
        info_dict = {}
        for info in t_infos:
            shard = info.shape
            zero = torch.zeros_like(info.tensor)
            if shard not in info_dict:
                info_dict[shard] = info
            elif torch.allclose(info_dict[shard].tensor, info.tensor):
                # clog.info(f"Multiple tensors for the same shard."
                #           f"Same value, abandon one: {info.canonical_name}")
                continue
            elif torch.allclose(info.tensor, zero):
                clog.info(f"Multiple tensors for the same shard. "
                          f"One is zero, abandon it: {info.canonical_name}")
            elif torch.allclose(info_dict[shard].tensor, zero):
                clog.info(f"Multiple tensors for the same shard. "
                          f"One is zero, abandon it: {info.canonical_name}")
                info_dict[shard] = info
            else: # Same shard, different grads, shouldn't happen. But add them.
                # TODO: check if need finalize grad (for PP, embedding)
                print(info_dict[shard].tensor.to(torch.float64).sum())
                print(info.tensor.to(torch.float64).sum())
                clog.warning(f"Multiple tensors for the same shard. "
                             f"Different value, add them: {info.canonical_name}")
                info_dict[shard].tensor += info.tensor
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
    
    @staticmethod
    def add_tensor_info(t_infos: list["TensorInfo"]) -> "TensorInfo":
        """
        Add the list of TensorInfo (all full shape) into one TensorInfo.
        Use tree-based reduction to minimize numerical error.
        """
        assert all(t.shape.is_full() for t in t_infos), \
            "Must finalize(add) full tensors."
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
            shape=t_infos[0].shape,
            module_class=t_infos[0].module_class,
            is_grad=t_infos[0].is_grad,
        )


T_TYPE = ("activ", "g_activ", "g_param", "param", "g_param_accum")


__all__ = ["TensorInfo", "T_TYPE"]
