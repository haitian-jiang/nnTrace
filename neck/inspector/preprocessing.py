import torch
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional
from loguru import logger as clog
from neck.structs import TensorInfo, T_TYPE


def _is_full_tensor(tensor_info: TensorInfo) -> bool:
    """
    Check if the tensor is full.
    """
    full_shape = tensor_info.shape.full_shape
    shard_slice = tensor_info.shape.shard_slice
    for dim, size in enumerate(full_shape):
        if shard_slice[dim] != slice(0, size):
            return False
    return True


def _merge_logs(
        log_collections: Dict[str, List[TensorInfo]]
    ) -> Dict[str, TensorInfo | List[TensorInfo]]:
    """
    Update the merged logs with the raw logs.
    """
    merged_logs = {}
    for c_name, log_shards in log_collections.items():
        if len(log_shards) == 1:
            merged_logs[c_name] = log_shards[0]
            continue
        if _is_full_tensor(log_shards[0]):  # list of tensor, should be all same
            for info in log_shards[1:]:
                assert _is_full_tensor(info)
                if not torch.allclose(log_shards[0].tensor, info.tensor):
                    clog.warning(f"Conflicting full tensors: {c_name}")
                    print(log_shards[0].tensor.to(torch.float64).sum())
                    print(info.tensor.to(torch.float64).sum())
            merged_logs[c_name] = log_shards[0]
        else:
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
    
    log_collections = defaultdict(list)
    canonical_names = set()  # just for checking
    for raw_log_file in raw_log_files:  # e.g.: rank0.pt
        raw_logs = torch.load(raw_log_file)
        for c_name, log in raw_logs.items():
            log_collections[c_name].append(log)
        canonical_names.update(raw_logs.keys())
    merged_logs = _merge_logs(log_collections)
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
