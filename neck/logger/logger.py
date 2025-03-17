import atexit
import torch
import datetime
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict
from omegaconf import OmegaConf
from loguru import logger as clog
from neck import CONFIG
from neck.canonical import get_parallel_states, trim_word_embeddings
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
    torch.save(LOGS[t_type], save_file)

    clog.debug(f"[Rank {rank}] 📝 Logged {t_type} tensors saved to {save_file}")


def _exit_hook():
    try:
        rank = get_parallel_states().rank.world_rank
        if not CONFIG.logging.enable_logging:
            if rank == 0:
                clog.debug("⛔️ Logging is disabled.")
                return
        world_size = get_parallel_states().size.world_size
        if (run_name:=CONFIG.logging.disk.get("run_name", None)) is None:
            time_name = datetime.datetime.now().strftime("%y-%m-%d-%H-%M")
            suffix = "single" if world_size == 1 else "distr"
            run_name = f"{time_name}-{suffix}"
        run_dir = Path(CONFIG.logging.disk.log_path) / f"{run_name}"
        run_dir.mkdir(parents=True, exist_ok=True)
        for t_type in TType:
            # trim word embeddings to vocab_size for single GPU
            if world_size == 1:
                trim_word_embeddings(LOGS[t_type])
            save_logs(t_type, run_dir)
    except AssertionError:
        # in get_parallel_states, torch.distributed is not initialized
        clog.debug("[Developing mode]: Bye!")


if CONFIG.logging.enable_logging:
    atexit.register(_exit_hook)
