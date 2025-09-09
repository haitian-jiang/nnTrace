import torch
from omegaconf import OmegaConf
from neck import CONFIG
from .megatron import get_megatron_lm_parallel_rank, get_megatron_parallel_options
from .vllm import get_vllm_parallel_rank, get_vllm_parallel_options
from .transformers import get_transformers_parallel_rank


GOT_PARALLEL_STATES = False


def get_parallel_states(device_mesh=None) -> OmegaConf:
    """
    Get the parallel rank of the current process, and store in 
    CONFIG.parallel.rank.{tp,dp,pp}_rank
    Get the sizes of the current run, and store in
    CONFIG.parallel.size.{world,{tp,dp,pp}{,_group}}_size
    """

    if "rank" in CONFIG.parallel and "size" in CONFIG.parallel and \
        CONFIG.parallel.rank and CONFIG.parallel.size:
        # Already computed
        return CONFIG.parallel
    
    CONFIG.parallel.rank = {}
    CONFIG.parallel.size = {}

    if CONFIG.parallel.framework == "Megatron-LM":
        results = get_megatron_lm_parallel_rank()
        get_megatron_parallel_options()
    elif CONFIG.parallel.framework == "vLLM":
        results = get_vllm_parallel_rank()
        get_vllm_parallel_options()
    elif CONFIG.parallel.framework == "transformers":
        results = get_transformers_parallel_rank(device_mesh)
    else:
        raise NotImplementedError(
            f"Parallel framework {CONFIG.parallel.framework} is not supported."
        )
    global GOT_PARALLEL_STATES
    GOT_PARALLEL_STATES = True
    return results


__all__ = [
    "GOT_PARALLEL_STATES",
    "get_parallel_states",
]
