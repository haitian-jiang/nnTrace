import torch
import torch.nn as nn
import neck.canonical as canonical
import neck.random
from neck.logger import log_tensor
from neck import CONFIG, TensorInfo
from typing import Optional
from .annotation import get_annotation


def reinit_model(
        model: torch.nn.Module, 
        vp_rank: Optional[int] = None,
        log_params: bool = True,
    ) -> None:
    """
    Reinitialize the model parameter tensors to be consistent.  
    Virtual pipeline rank cannot be inferred, so it must be provided.
    If virtual pipeline is not used, vp_rank should be None.
    """
    assert isinstance(model, nn.Module), "model must be a torch.nn.Module"
    canonical.get_parallel_states()  
    if CONFIG.parallel.size.vp_size is None:
        assert vp_rank is None or vp_rank ==0, \
            "vp_rank must be None or 0 if virtual pipeline is not used"
        vp_rank = None  # for compatibility with downstream code
    else:
        assert vp_rank is not None, \
            "vp_rank must be provided if virtual pipeline is used"
        assert vp_rank < CONFIG.parallel.size.vp_size, \
            "vp_rank must be less than the virtual pipeline size"
    
    # check that the model has already been initialized
    assert (
        any(param.numel() > 0 for param in model.parameters()) and \
        hasattr(model, "state_dict")
    ), "model has not been initialized"

    prefix = canonical.get_prefix(set(("TP", "PP", "VP")), vp_rank)
    for name, param in model.named_parameters():
        local_name = f"{prefix}{name}"
        canonical_name = canonical.to_canonical_param_mod(name, vp_rank)
        annotation = get_annotation(local_name, "param")
        assert annotation is not None, \
            f"Annotation not found for {local_name}"
        full_shape, tp_slice = canonical.infer_tp_slice(param.shape, annotation)

        # reinitialize the parameter
        new_param = neck.random.rand(canonical_name, full_shape, tp_slice)
        new_param = new_param.to(param.dtype)
        param.data = new_param.data.to(param.device)

        if log_params:
            tensor_info = TensorInfo(
                tensor=new_param,
                canonical_name=canonical_name,
                full_shape=full_shape,
                shard_slice=tp_slice,
                local_name=local_name
            )
            log_tensor(canonical_name, "param", tensor_info)