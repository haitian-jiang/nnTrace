import math
import torch
import torch.nn as nn
import neck.random
from neck import CONFIG
from typing import Optional
from neck.structs import (
    get_annotation, 
    TensorInfo, 
    TensorAnnotation,
    ModuleAnnotation,
    TType, DTYPE, MCH_EPS,
)
from neck.canonical import (
    get_parallel_states,
    to_canonical_param_mod, 
    infer_slice,
)
from neck.logger import (
    log_tensor, 
    register_hook, 
    regist_batch_cnt_hook, 
    get_param_full_cname,
    ParamLogHook,
    PerturbationHook,
)


def _check_vp_rank(
    model: torch.nn.Module,
    vp_rank: Optional[int] = None,
):
    """
    Check the model and virtual pipeline rank are valid.
    Used for reinit_model and register_model_hooks.
    """
    assert isinstance(model, nn.Module), "model must be a torch.nn.Module"
    get_parallel_states()
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

    for name, param in model.named_parameters():
        # canonical parameter of this framework, need to convert to fw agnostic
        fw_canon_param_name = to_canonical_param_mod(name, vp_rank)
        annotation: TensorAnnotation = get_annotation(name, "param")
        assert annotation is not None, \
            f"Annotation not found for {name}"
        # get framework agnostic canon_param_name
        # currently only supports replacement for Megatron and TransformerEngine
        # TODO: can have a separate function to handle this and support regex
        if annotation.canonical_suffix is not None:
            canon_param_name = fw_canon_param_name.replace(
                annotation.suffix, annotation.canonical_suffix
            )
        else:  
            canon_param_name = fw_canon_param_name
            fw_canon_param_name = None  # work as arg for ParamLogHook
        shard_slice = infer_slice(param.shape, annotation)

        # "iter.0|module.module.decoder.layers.0.xxx|param"        
        full_cname = get_param_full_cname(canon_param_name, TType.param)

        # reinitialize the parameter
        new_param = neck.random.rand(full_cname, shard_slice)
        # respect the original range
        if initialization == "kaiming_uniform":
            new_param = (2 * new_param - 1) / math.sqrt(shard_slice.full_shape[-1])
        # TODO: other initialization

        new_param = new_param.to(param.dtype)
        if log_params:
            tensor_info = TensorInfo(
                tensor=new_param,
                canonical_name=full_cname,
                shard_slice=shard_slice,
            )
            log_tensor(full_cname, TType.param, tensor_info)

        new_param = new_param.to(param.device)
        param.data.copy_(new_param.data)


        param.register_hook(
            ParamLogHook(
                canon_param_name,  # tensor bwd hook compose its own name
                vp_rank,
                shard_slice,
                fw_canon_param_name,  # None if not changed
            )
        )


def filter_module_annotation(m_annotation: ModuleAnnotation) -> ModuleAnnotation:
    """
    Filter out the unnecessary fields parallel dim in the module annotation 
    according to the opened parallel options.
    Currently only sequence parallel dim info.

    Args:
        m_annotation: The module annotation to be filtered.

    Returns:
        The filtered module annotation.
    """
    assert m_annotation is not None, "m_annotation must be provided"
    if CONFIG.parallel.get("sequence_parallel", False) == False:
        if m_annotation.input:
            m_annotation.input.sp_dim = None
        if m_annotation.output:
            m_annotation.output.sp_dim = None
    return m_annotation


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
        m_annotation: ModuleAnnotation = get_annotation(m_name, "module")
        if m_annotation is None or m_annotation.turnoff:
            continue
        m_annotation = filter_module_annotation(m_annotation)
        canon_mod_name = to_canonical_param_mod(m_name, vp_rank)
        register_hook(module, canon_mod_name, m_annotation, vp_rank)

    regist_batch_cnt_hook(model, vp_rank, True)  # prepend
    regist_batch_cnt_hook(model, vp_rank, False)  # prepend


def register_perturbation_hook(
    module: torch.nn.Module,
    scale: float = 0.25,
    dtype: str = "bf16",
    tensor_id: Optional[str] = None,
):
    """
    Register a hook to perturb the model input to estimate the FP error.

    Args:
        model: The model to register the hook on.
        scale: The scale of the perturbation (times machine epsilon).
    """
    assert dtype in DTYPE, f"Unsupported dtype: {dtype}"
    mch_eps = MCH_EPS[dtype]
    perturb_hook = PerturbationHook(scale, mch_eps, tensor_id)
    module.register_forward_hook(perturb_hook)


__all__ = [
    "reinit_model",
    "register_model_hooks",
]
