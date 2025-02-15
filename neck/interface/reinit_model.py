import math
import torch
import torch.nn as nn
import neck.canonical as canonical
import neck.random
from loguru import logger as clog
from neck.logger import log_tensor, register_hook
from neck import CONFIG, TensorInfo
from typing import Optional
from .annotation import get_annotation


def _check_vp_rank(
        model: torch.nn.Module,
        vp_rank: Optional[int] = None,
    ):
    """
    Check the model and virtual pipeline rank are valid.
    Used for reinit_model and register_model_hooks.
    """
    assert isinstance(model, nn.Module), "model must be a torch.nn.Module"
    canonical.get_parallel_states()
    if CONFIG.parallel.size.vp_size is None:
        assert vp_rank is None or vp_rank ==0, \
            "vp_rank must be None or 0 if virtual pipeline is not used"
        vp_rank = 0  # rely on CONFIG.parallel.size.vp_size to show no vpp
    else:
        assert vp_rank is not None, \
            "vp_rank must be provided if virtual pipeline is used"
        assert vp_rank < CONFIG.parallel.size.vp_size, \
            "vp_rank must be less than the virtual pipeline size"
    return vp_rank


def reinit_model(
        model: torch.nn.Module, 
        vp_rank: Optional[int] = None,
        log_params: bool = True,
        initialization: str = "kaiming_uniform",
    ) -> None:
    """
    Reinitialize the model parameter tensors to be consistent.  
    Virtual pipeline rank cannot be inferred, so it must be provided.
    If virtual pipeline is not used, vp_rank should be None.
    """
    vp_rank = _check_vp_rank(model, vp_rank)
    assert isinstance(vp_rank, int), "vp_rank must be an integer"
    
    # check that the model has already been initialized
    assert (
        any(param.numel() > 0 for param in model.parameters()) and \
        hasattr(model, "state_dict")
    ), "model has not been initialized"

    prefix = canonical.get_prefix(set(("TP", "PP", "VP")), vp_rank)
    for name, param in model.named_parameters():
        local_name = f"{prefix}{name}"
        canon_param_name = canonical.to_canonical_param_mod(name, vp_rank)
        annotation = get_annotation(local_name, "param")
        assert annotation is not None, \
            f"Annotation not found for {local_name}"
        full_shape, tp_slice = canonical.infer_tp_slice(param.shape, annotation)

        # reinitialize the parameter
        new_param = neck.random.rand(canon_param_name, full_shape, tp_slice)
        # respect the original range
        if initialization == "kaiming_uniform":
            new_param = (2 * new_param - 1) / math.sqrt(full_shape[-1])
        new_param = new_param.to(param.dtype)
        param.data = new_param.data.to(param.device)

        if log_params:
            tensor_info = TensorInfo(
                tensor=new_param,
                canonical_name=canon_param_name,
                full_shape=full_shape,
                shard_slice=tp_slice,
                local_name=local_name
            )
            log_tensor(canon_param_name, "param", tensor_info)


def register_model_hooks(
        model: torch.nn.Module,
        vp_rank: Optional[int] = None,

    ):
    """
    Register hooks to log activations / grads during forward / backward pass.
    """
    vp_rank = _check_vp_rank(model, vp_rank)
    assert isinstance(vp_rank, int), "vp_rank must be an integer"

    for m_name, module in model.named_modules():
        m_annotation = get_annotation(m_name, "module")
        if m_annotation is None:
            continue
        canon_mod_name = canonical.to_canonical_param_mod(m_name, vp_rank)
        rank = CONFIG.parallel.rank.world_rank
        clog.debug(f"[Rank {rank}] Registering hooks for {canon_mod_name}")
        register_hook(module, canon_mod_name, m_annotation)
