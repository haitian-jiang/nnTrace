import torch
import torch.nn as nn
import neck.canonical as canonical
import neck.random
from neck.canonical import REG_PARAM
from neck import CONFIG, TensorInfo
from typing import Optional


def reinit_model(
        model: torch.nn.Module, 
        vp_rank: Optional[int] = None
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
        full_shape, tp_slice = canonical.get_tp_slice(name, param)
        tensor_info = TensorInfo(
            tensor_id=id(param),
            canonical_name=canonical_name,
            full_shape=full_shape,
            shard_slice=tp_slice,
            dtype=param.dtype,
            local_name=local_name
        )
        canonical.register_tensor(param, tensor_info, REG_PARAM)

        # reinitialize the parameter
        new_param = neck.random.rand(canonical_name, full_shape, tp_slice)
        param.data = new_param.data.to(param.dtype).to(param.device)