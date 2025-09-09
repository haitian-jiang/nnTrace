import os
import atexit
import torch
import datetime
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict
from omegaconf import OmegaConf
from loguru import logger as clog
from neck import CONFIG
from neck.canonical import get_parallel_states
from neck.structs import TensorInfo, TType
from .log_hook import LOG_HOOKS


LOGS = defaultdict(dict)


def log_tensor(
    canonical_name: str,
    t_type: TType,
    tensor_info: TensorInfo,
) -> None:
    """
    Log the tensor to the destination.
    """
    if not CONFIG.logging.enable_logging:
        return
    assert tensor_info.tensor.device.type == "cpu", \
        "Tensors need to be moved to CPU before being logged"

    for hook in LOG_HOOKS:
        tensor_info = hook(tensor_info, t_type)

    destination = LOGS[t_type]
    if canonical_name in destination:
        old_tensor = destination[canonical_name].tensor
        if torch.allclose(old_tensor, tensor_info.tensor):
            return
        else:
            clog.error(f"{t_type}: Conflicting tensors when logging: {canonical_name}")
    destination[canonical_name] = tensor_info


def clear_logs(*args: TType) -> None:
    for t_type in args:
        if t_type in LOGS:
            LOGS[t_type].clear()
    clog.debug(f"Cleared logs for {', '.join(map(str, args))}")


def save_logs(
    t_type: TType, 
    run_dir: Path,
) -> None:
    OmegaConf.save(CONFIG, run_dir / "config.yaml")

    world_size = get_parallel_states().size.world_size
    rank = get_parallel_states().rank.world_rank
    if world_size == 1:
        save_file = run_dir / f"{t_type}.pt"
    else:
        p = run_dir / t_type
        p.mkdir(parents=True, exist_ok=True)
        save_file = p / f"rank{rank}.pt"
    if len(LOGS[t_type]) == 0:
        clog.warning(f"[Rank {rank}] ⚠️ Empty logs for {t_type}, not saving. "
                     f"Killed or training skipped due to existing checkpoints.")
        return
    torch.save(LOGS[t_type], save_file)

    clog.debug(f"[Rank {rank}] 📝 Logged {t_type} tensors saved to {save_file}")


def post_processing_before_save(LOGS):
    if CONFIG.parallel.framework == "Megatron-LM":
        from neck.canonical.megatron import post_processing_before_save_megatron
        post_processing_before_save_megatron(LOGS)
    elif CONFIG.parallel.framework == "vLLM":
        from neck.canonical.vllm import post_processing_before_save_vllm
        post_processing_before_save_vllm(LOGS)
    elif CONFIG.parallel.framework == "transformers":
        from neck.canonical.transformers import post_processing_before_save_transformers
        post_processing_before_save_transformers(LOGS)
    else:
        raise NotImplementedError(
            f"Parallel framework {CONFIG.parallel.framework} is not supported."
        )


def _exit_hook():
    try:
        rank = get_parallel_states().rank.world_rank
        if not CONFIG.logging.enable_logging:
            if rank == 0:
                clog.debug("⛔️ Logging is disabled.")
                return
        global LOGS
        post_processing_before_save(LOGS)

        world_size = get_parallel_states().size.world_size
        if not (run_name:=os.getenv("NECK_RUN_NAME", None)):  # not ("" or None)
            if (run_name:=CONFIG.logging.disk.get("run_name", None)) is None:
                time_name = datetime.datetime.now().strftime("%y-%m-%d-%H-%M")
                suffix = "single" if world_size == 1 else "distr"
                run_name = f"{time_name}-{suffix}"
        run_dir = Path(CONFIG.logging.disk.log_path) / f"{run_name}"
        run_dir.mkdir(parents=True, exist_ok=True)
        for t_type in TType:
        #     # trim word embeddings to vocab_size for single GPU
        #     if world_size == 1:
        #         trim_word_embeddings(LOGS[t_type])
            save_logs(t_type, run_dir)
    except AssertionError:
        # in get_parallel_states, torch.distributed is not initialized
        clog.debug("[Developing mode]: Bye!")


def save_all_logs_before_exit():
    """
    Call the exit hook to save all logs before exiting and unregister the hook.
    """
    _exit_hook()
    atexit.unregister(_exit_hook)


if CONFIG.logging.enable_logging and CONFIG.logging.save_logs_atexit:
    atexit.register(_exit_hook)


__all__ = [
    "LOGS",
    "log_tensor",
    "clear_logs",
    "save_logs",
    "save_all_logs_before_exit",
]
