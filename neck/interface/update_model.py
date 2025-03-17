import torch
from typing import Optional
from .initialize_model import _check_vp_rank
from neck.structs import get_annotation, TensorInfo, TType
from neck.canonical import to_canonical_param_mod, infer_slice
from neck.logger import log_tensor, update_iter_batch_cnt, get_param_logging_cname


def prior_optim_logging(
        model: torch.nn.Module,
        vp_rank: Optional[int] = None,
    ) -> None:
    """
    Log the main gradient of the model parameters before the optimizer step.
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
        canon_param_logging_name = get_param_logging_cname(canon_param_name)
        tensor_info = TensorInfo(
            main_grad,
            canon_param_logging_name,
            shard_slice,
            is_grad=True,
        )
        log_tensor(canon_param_logging_name, TType.g_param_main, tensor_info)


def post_optim_logging_reset(
        model: torch.nn.Module,
        vp_rank: Optional[int] = None,    
    ) -> None:
    """
    Log the new parameters after weight update to check whether there are bugs
    in the optimizer. Best if lr is large (like 1 or larger), just for checking.
    Then reset the batch counter for a new iteration, and increase iter counter.
    """
    vp_rank = _check_vp_rank(model, vp_rank)
    assert isinstance(vp_rank, int), "vp_rank must be an integer"

    # Reset the iter and batch counter for a new iteration
    # NOTE: need to update the cnt first because will be reflected in the name
    update_iter_batch_cnt()

    for name, param in model.named_parameters():
        canon_param_name = to_canonical_param_mod(name, vp_rank)
        annotation = get_annotation(name, "param")
        assert annotation is not None, \
            f"Annotation not found for {name}"
        shard_slice = infer_slice(param.shape, annotation)
        param_data = param.data.detach().cpu()
        canon_param_logging_name = get_param_logging_cname(canon_param_name)
        tensor_info = TensorInfo(
            param_data,
            canon_param_logging_name,
            shard_slice,
            is_grad=False,
        )
        log_tensor(canon_param_logging_name, TType.param, tensor_info)
        # TODO: optionally rewrite the param.data with the consistent value
