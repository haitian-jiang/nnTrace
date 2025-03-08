import torch
import neck.canonical as canonical
from typing import Optional
from .reinit_model import _check_vp_rank
from neck.structs import get_annotation, TensorInfo, ShardNDSlices, TType
from neck.logger import log_tensor


def log_main_grad(
        model: torch.nn.Module,
        vp_rank: Optional[int] = None,
    ) -> None:
    """
    Log the main gradient of the model parameters before the optimizer step.
    """
    
    vp_rank = _check_vp_rank(model, vp_rank)
    assert isinstance(vp_rank, int), "vp_rank must be an integer"

    for name, param in model.named_parameters():
        canon_param_name = canonical.to_canonical_param_mod(name, vp_rank)
        annotation = get_annotation(name, "param")
        assert annotation is not None, \
            f"Annotation not found for {name}"
        shard_slice = canonical.infer_tp_slice(param.shape, annotation)
        main_grad = param.main_grad.detach().cpu()  # for Megatron
        tensor_info = TensorInfo(
            main_grad,
            canon_param_name,
            shard_slice,
            is_grad=True,
        )
        log_tensor(canon_param_name, TType.g_param_main, tensor_info)
