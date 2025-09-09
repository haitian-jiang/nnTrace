import torch
from collections import defaultdict
from functools import reduce
from omegaconf import OmegaConf
from neck import CONFIG
from neck.structs import TType, TensorInfo, NDSlice

def get_vllm_parallel_rank() -> OmegaConf:
    from vllm.distributed import parallel_state
    if hasattr(parallel_state, "model_parallel_is_initialized"):
        is_initialized = parallel_state.model_parallel_is_initialized
    else:
        raise AttributeError("Cannot find is_initialized in parallel_state.")
    assert is_initialized(), "vLLM parallel state is not initialized."
    
    rank = CONFIG.parallel.rank
    size = CONFIG.parallel.size

    rank.world_rank = torch.distributed.get_rank()
    rank.dp_rank = parallel_state.get_dp_group().rank_in_group
    rank.tp_rank = parallel_state.get_tensor_model_parallel_rank()
    rank.pp_rank = parallel_state.get_pp_group().rank_in_group

    size.world_size = torch.distributed.get_world_size()
    size.dp_size = parallel_state.get_dp_group().world_size
    size.tp_size = parallel_state.get_tensor_model_parallel_world_size()
    size.pp_size = parallel_state.get_pp_group().world_size
    size.vp_size = None  # vLLM does not use virtual pipeline parallelism

    return CONFIG.parallel


def get_vllm_parallel_options() -> None:
    CONFIG.parallel.types = []
    sizes = CONFIG.parallel.size
    for xp in ["tp", "dp", "pp"]:
        if sizes.get(f"{xp}_size", 1) > 1:
            CONFIG.parallel.types.append(xp)


def post_processing_before_save_vllm(LOGS: dict[TType, dict[str, TensorInfo]]) -> None:
    assert len(LOGS) == 1 and TType.activ in LOGS, \
        "vLLM should only record activations."
    activ = LOGS[TType.activ]
    temp_activ_dict: dict[str, dict[int, TensorInfo]] = defaultdict(dict)
    for cname, info in activ.items():
        # iter.6|batch.-1|layers.7.self_attn.attn|activ.input.2
        it, batch, layer, attrib = cname.split("|")
        if layer.endswith("self_attn.attn") and attrib.startswith("activ.input."):
            attrib = attrib.replace("activ.input.", "")
            attrib = ["Q", "K", "V"][int(attrib)]
        it = int(it.split(".")[1])  # iter.6 -> 6
        name = f"{batch}|{layer}|{attrib}"
        temp_activ_dict[name][it] = info
    
    max_steps = [max(d.keys()) for d in temp_activ_dict.values()]
    assert all(s == max_steps[0] for s in max_steps), \
        "All layers should have the same number of iterations."
    max_steps = max_steps[0]

    temp_activ_list: dict[str, list[TensorInfo]] = defaultdict(lambda: [None]*(max_steps+1))
    for name, info_dict in temp_activ_dict.items():
        for it, info in info_dict.items():
            temp_activ_list[name][it] = info

    new_activ = {}
    seq_len = 0
    for name, info_list in temp_activ_list.items():
        start_idx = 0
        prefill: torch.Tensor = info_list[0].tensor
        if seq_len == 0:
            seq_dim = 0
            seq_len = sum([info.tensor.shape[seq_dim] for info in info_list])

        concat_shape = list(prefill.shape)
        concat_shape[seq_dim] = seq_len
        concat_tensor = torch.zeros(concat_shape, dtype=prefill.dtype)
        for info in info_list:
            s = info.tensor.shape[seq_dim]
            concat_tensor[start_idx:start_idx + s] = info.tensor
            start_idx += s
        new_activ[name] = concat_tensor
    LOGS[TType.activ] = new_activ


