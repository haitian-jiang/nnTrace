import math
import torch
from typing import Optional
from .initialize_model import _check_vp_rank
from neck.structs import get_annotation, TensorInfo, TType
from neck.canonical import to_canonical_param_mod, infer_slice
from neck.logger import log_tensor, update_iter_batch_cnt, get_param_full_cname
import neck.random


def prior_optim(
    model: torch.nn.Module,
    vp_rank: Optional[int] = None,
) -> None:
    """
    Log the main gradient of the model parameters before the optimizer step.
    Rewrite the main grad to eliminate the errors to test the optimizer.
    """
    
    vp_rank = _check_vp_rank(model, vp_rank)
    assert isinstance(vp_rank, int), "vp_rank must be an integer"

    for name, param in model.named_parameters():
        canon_param_name = to_canonical_param_mod(name, vp_rank)
        annotation = get_annotation(name, "param")
        assert annotation is not None, \
            f"Annotation not found for {name}"
        shard_slice = infer_slice(param.shape, annotation)
        main_grad = param.main_grad.detach().cpu()  # for Megatron
        # full cname is iter+cname+t_type
        full_cname = get_param_full_cname(canon_param_name, TType.g_param_main)
        tensor_info = TensorInfo(
            main_grad,
            full_cname,
            shard_slice,
            is_grad=True,
        )
        log_tensor(full_cname, TType.g_param_main, tensor_info)

        # rewrite the main_grad tensor
        # new_main_grad = neck.random.rand(full_cname, shard_slice)
        # new_main_grad = (new_main_grad - 0.5) / math.sqrt(shard_slice.full_shape[-1])  # kaiming uniform, but half the range
        # new_main_grad = new_main_grad.to(param.dtype)
        # param.main_grad.data = new_main_grad.data.to(param.device)



def post_optim(
    model: torch.nn.Module,
    vp_rank: Optional[int] = None,    
) -> None:
    """
    Reset the batch counter for a new iteration, and increase iter counter.
    Log the new parameters after weight update to check whether there are bugs
    in the optimizer. Best if lr is large (like 1 or larger), just for checking.
    """
    vp_rank = _check_vp_rank(model, vp_rank)
    assert isinstance(vp_rank, int), "vp_rank must be an integer"

    # Reset the iter and batch counter for a new iteration
    # NOTE: need to update the cnt first because will be reflected in the name
    update_iter_batch_cnt()

    for name, param in model.named_parameters():
        fw_canon_param_name = to_canonical_param_mod(name, vp_rank)
        annotation = get_annotation(name, "param")
        assert annotation is not None, \
            f"Annotation not found for {name}"
        # same as `reinit_model`, get the framework agnostic canon_param_name
        if annotation.canonical_suffix is not None:
            canon_param_name = fw_canon_param_name.replace(
                annotation.suffix, annotation.canonical_suffix
            )
        else:  
            canon_param_name = fw_canon_param_name
            fw_canon_param_name = None  # work as arg for ParamLogHook
        shard_slice = infer_slice(param.shape, annotation)
        param_data = param.data.detach().cpu()
        full_cname = get_param_full_cname(canon_param_name, TType.param)
        tensor_info = TensorInfo(
            param_data,
            full_cname,
            shard_slice,
        )
        log_tensor(full_cname, TType.param, tensor_info)
        # TODO: optionally rewrite the param.data with the consistent value


__all__ = [
    "prior_optim",
    "post_optim",
]
