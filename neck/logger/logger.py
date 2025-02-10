import atexit
import torch
import pathlib
import datetime
from collections import defaultdict
from typing import Dict, Optional
from neck import CONFIG
from neck.canonical import get_parallel_states


LOGS = defaultdict(dict)
T_TYPE = ("fwd_activ", "bwd_activ", "bwd_param", "param")
LOGS["fwd_activ"] = {"haha": torch.tensor(1)}


def log_tensor(
        name: str, 
        t_type: str,
        tensor: Optional[torch.Tensor], 
        info: Optional[Dict] = None, # TODO: type
    ) -> None:
    """
    Log the tensor to the destination.
    """
    if not CONFIG.logging.enable_logging or tensor is None:
        return
    assert t_type in T_TYPE, f"Invalid tensor type {t_type}"
    destination = LOGS[t_type]
    assert name not in destination, f"{name} already exists in the logger"
    # TODO: TensorInfo for params.
    # TODO: Module info for activations.
    destination[name] = tensor.to('cpu')


def save_logs(t_type: str, date: str):
    assert t_type in T_TYPE, f"Invalid tensor type {t_type}"

    paths = CONFIG.logging.disk
    match t_type:
        case "fwd_activ":
            directory = paths.log_path_fwd
        case "bwd_activ":
            directory = paths.log_path_bwd
        case "bwd_param":
            directory = paths.log_path_bwd
        case "param":
            directory = paths.log_path_param
    directory = pathlib.Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    world_size = get_parallel_states().size.world_size
    rank = get_parallel_states().rank.world_rank
    if world_size == 1:
        save_file = directory / f"{date}-single.pt"
    else:
        p = directory / date
        p.mkdir(parents=True, exist_ok=True)
        save_file = p / f"{rank}.pt"
    torch.save(LOGS[t_type], save_file)

    print(f"[Rank {rank}] 📝 Logged {t_type} tensors saved to {save_file}")


def _exit_hook():
    try:
        rank = get_parallel_states().rank.world_rank
        date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M")
        if not CONFIG.logging.enable_logging:
            if rank == 0:
                print("⛔️ Logging is disabled.")
        else:
            for t_type in T_TYPE:
                save_logs(t_type, date)
    except AssertionError:
        print("[Developing mode]: Bye!")


if CONFIG.logging.enable_logging:
    atexit.register(_exit_hook)