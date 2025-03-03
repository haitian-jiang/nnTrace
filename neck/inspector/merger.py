import torch
from collections import defaultdict
from pathlib import Path
from typing import Optional
from loguru import logger as clog
from neck.structs import TensorInfo, T_TYPE


def _merge_logs(
        log_collections: dict[str, list[TensorInfo]],
        t_type: str
    ) -> dict[str, TensorInfo | list[TensorInfo]]:
    """
    Update the merged logs with the raw logs.
    """
    merged_logs: dict[str, TensorInfo | list[TensorInfo]] = {}
    for c_name, log_shards in log_collections.items():
        if len(log_shards) == 1:
            merged_logs[c_name] = log_shards[0]
            continue
        # --- below deals with list of tensors --- #
        if log_shards[0].shape.is_full():  # either same or need finalize grad
            # check if all are full tensors
            for info in log_shards[1:]:
                assert info.shape.is_full(), \
                    "Conflicting shapes for the same canonical name."
            # check if the full tensors are the same
            all_same = all(
                torch.allclose(log_shards[0].tensor, info.tensor) 
                for info in log_shards[1:]
            )
            if all_same:
                merged_logs[c_name] = log_shards[0]
            # check if need finalize grad
            elif (
                t_type == "g_param" and
                all(
                    info.extra.get("need_finalize_grad", False)
                    for info in log_shards
                )
            ):
                merged_logs[c_name] = TensorInfo.add_tensor_info(log_shards)
            else:
                clog.error(f"Conflicting full tensors: {c_name}, not merged.")
        else:  # shard tensors
            merged_logs[c_name] = TensorInfo.merge_tensor_info(log_shards)
    return merged_logs
        

def _merge_distr_logs(log_path: str, t_type: str) -> None:
    """
    Merge the logs from different ranks into a single log file.
    Works for one specific tensor type.
    """
    merged_file_path = Path(log_path) / f"{t_type}.pt"
    if merged_file_path.exists():
        return
    
    raw_log_dir = Path(log_path) / t_type
    raw_log_files = raw_log_dir.glob("*.pt")
    clog.info(f"Merging logs from {raw_log_dir}")
    
    log_collections: dict[str, list[TensorInfo]] = defaultdict(list)
    canonical_names: set[str] = set()  # just for checking
    for raw_log_file in raw_log_files:  # e.g.: rank0.pt
        raw_logs: dict[str, TensorInfo] = torch.load(raw_log_file)
        for c_name, log in raw_logs.items():
            log_collections[c_name].append(log)
        canonical_names.update(raw_logs.keys())
    merged_logs = _merge_logs(log_collections, t_type)
    for canonical_name in canonical_names:
        assert canonical_name in merged_logs
    torch.save(merged_logs, merged_file_path)
    clog.info(f"Saved merged logs to {merged_file_path}")


def merge_distr_logs(log_path: str, t_type: Optional[str] = None) -> None:
    """
    Merge the logs from different ranks into a single log file.
    Works for one specific tensor type.
    """
    if t_type is not None:
        assert t_type in T_TYPE, f"Invalid tensor type: {t_type}"
        _merge_distr_logs(log_path, t_type)
    else:
        for t_type in T_TYPE:
            _merge_distr_logs(log_path, t_type)
