import torch
from omegaconf import OmegaConf
from typing import Tuple, Optional
from neck import CONFIG
from .megatron import get_megatron_lm_parallel_rank


def get_parallel_states() -> OmegaConf:
    """
    Get the parallel rank of the current process, and store in 
    CONFIG.parallel.rank.{tp,dp,pp}_rank
    Get the sizes of the current run, and store in
    CONFIG.parallel.size.{world,{tp,dp,pp}{,_group}}_size
    """
    assert torch.distributed.is_initialized(), \
        "Distributed environment is not initialized."

    if "rank" in CONFIG.parallel and "size" in CONFIG.parallel and \
        CONFIG.parallel.rank and CONFIG.parallel.size:
        # Already computed
        return CONFIG.parallel
    
    CONFIG.parallel.rank = {}
    CONFIG.parallel.size = {}

    if CONFIG.parallel.framework == "Megatron-LM":
        return get_megatron_lm_parallel_rank()
    else:
        raise NotImplementedError(
            f"Parallel framework {CONFIG.parallel.framework} is not supported."
        )


def get_prefix(
        parallel_dim: set = None, 
        vp_rank: Optional[int] = None
    ) -> str:
    if parallel_dim is None:
        raise ValueError("parallel_dim must be provided.")
    
    for strategy in parallel_dim:
        if strategy not in ["TP", "DP", "PP", "VP"]:
            raise ValueError(f"Invalid parallel strategy {strategy}.")
        
    para_conf = get_parallel_states()
    prefix = ""
    if "TP" in parallel_dim and para_conf.size.tp_size and para_conf.size.tp_size > 1:
        prefix += f"TP{para_conf.rank.tp_rank}/{para_conf.size.tp_size}|"
    if "DP" in parallel_dim and para_conf.size.dp_size and para_conf.size.dp_size > 1:
        prefix += f"DP{para_conf.rank.dp_rank}/{para_conf.size.dp_size}|"
    if "PP" in parallel_dim and para_conf.size.pp_size and para_conf.size.pp_size > 1:
        prefix += f"PP{para_conf.rank.pp_rank}/{para_conf.size.pp_size}|"
    if "VP" in parallel_dim and para_conf.size.vp_size and para_conf.size.vp_size > 1:
        prefix += f"VP{vp_rank}/{para_conf.size.vp_size}|"
    
    return prefix