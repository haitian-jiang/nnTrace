import math
import torch
from typing import Optional
from .initialize_model import _check_vp_rank
from neck.structs import get_annotation, TensorInfo, TType
from neck.canonical import to_canonical_param_mod, infer_slice
from neck.logger import log_tensor, get_param_full_cname
import neck.random


def prior_optim(
    model: torch.nn.Module,
    vp_rank: Optional[int] = None,
    log_main_grad: bool = True,
    rewrite_main_grad: Optional[str | int | float] = None,
) -> None:
    """
    Log the main gradient of the model parameters before the optimizer step.
    Optionally rewrite main_grad to eliminate the errors to test the optimizer.
    rewrite_main_grad: None (default) means no rewrite, "full" means fully 
    rewrite with random values, or a float/int to scale the main_grad by that 
    value to make the bug error more prominent if there is any.
    """
    
    vp_rank = _check_vp_rank(model, vp_rank)
    assert isinstance(vp_rank, int), "vp_rank must be an integer"

    for name, param in model.named_parameters():
        canon_param_name = to_canonical_param_mod(name, vp_rank)
        annotation = get_annotation(name, "param")
        # assert annotation is not None, \
        #     f"Annotation not found for {name}"
        if annotation is None:
            shard_slice = neck.structs.ShardNDSlices(param.shape)
        else:
            shard_slice = infer_slice(param, annotation)
        main_grad = param.main_grad.detach().cpu()  # for Megatron
        # full cname is iter+cname+t_type
        full_cname = get_param_full_cname(canon_param_name, TType.g_param_main)
        tensor_info = TensorInfo(
            main_grad,
            full_cname,
            shard_slice,
            is_grad=True,
        )
        
        # NOTE: currently not support checking main_grad for ZeRO, so can turn
        # off the logging for main_grad to avoid error
        if log_main_grad:  
            log_tensor(full_cname, TType.g_param_main, tensor_info)

        # NOTE: rewrite the main_grad tensor
        # This is used for testing the optimizer (e.g., DistribuedOptimizer)
        # It is useful for fp16 because the first run in fp16 with dynamic
        # loss scaler will have grad all being nan or inf, thus skipping updating.
        if rewrite_main_grad is not None and rewrite_main_grad == "full":
            new_main_grad = neck.random.rand(full_cname, shard_slice)
            # here we do not respect the original range, to execerbate the bug.
            new_main_grad = new_main_grad.to(param.dtype).to(param.device)
            param.main_grad.data.copy_(new_main_grad.data)
        elif rewrite_main_grad is not None and isinstance(rewrite_main_grad, (int, float)):
            param.main_grad.data.mul_(rewrite_main_grad)  # scale the main grad


def post_optim(
    model: torch.nn.Module,
    vp_rank: Optional[int] = None,
    rewrite_params: bool = False,
) -> None:
    """
    Reset the batch counter for a new iteration, and increase iter counter.
    Log the new parameters after weight update to check whether there are bugs
    in the optimizer. Best if lr is large (like 1 or larger), just for checking.
    """
    vp_rank = _check_vp_rank(model, vp_rank)
    assert isinstance(vp_rank, int), "vp_rank must be an integer"

    # NOTE: update_iter_batch_cnt() must be called before post_optim() so that
    # the iter and batch counter are correctly reflected in the parameter name

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
        shard_slice = infer_slice(param, annotation)
        param_data = param.data.detach().cpu()
        full_cname = get_param_full_cname(canon_param_name, TType.param)
        tensor_info = TensorInfo(
            param_data,
            full_cname,
            shard_slice,
        )
        log_tensor(full_cname, TType.param, tensor_info)
        # TODO: optionally rewrite the param.data with the consistent value
        if rewrite_params:
            iter_id = neck.logger.iter_cnt
            new_param = neck.random.rand(full_cname, shard_slice, iter_id)
            new_param = (2 * new_param - 1) / math.sqrt(shard_slice.full_shape[-1])
            new_param = new_param.to(param.dtype).to(param.device)
            param.data.copy_(new_param.data)



__all__ = [
    "prior_optim",
    "post_optim",
]
