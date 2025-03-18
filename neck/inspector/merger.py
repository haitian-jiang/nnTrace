import torch
from collections import defaultdict
from pathlib import Path
from typing import Optional
from loguru import logger as clog
from neck.structs import TensorInfo, TType
from neck.canonical import trim_word_embeddings
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm


def _merge_logs(
    c_name: str,
    log_shards: list[TensorInfo],
    t_type: TType,
) -> TensorInfo | None:
    """
    Update the merged logs with the raw logs.
    """
    if len(log_shards) == 1:
        return c_name, log_shards[0]

    # --- below deals with list of tensors --- #
    if log_shards[0].shard_slice.is_full():  # either same or need finalize grad
        # check if all are full tensors
        for info in log_shards[1:]:
            assert info.shard_slice.is_full(), \
                "Conflicting shapes for the same canonical name."
        # check if the full tensors are the same
        all_same = all(
            torch.allclose(log_shards[0].tensor, info.tensor) 
            for info in log_shards[1:]
        )
        if all_same:
            return c_name, log_shards[0]
        # check if need finalize grad
        elif (
            t_type is TType.g_param and
            all(
                info.extra.get("need_finalize_grad", False)
                for info in log_shards
            )
        ):
            return c_name, TensorInfo.add_tensor_info(log_shards)
        else:
            clog.error(f"{t_type}: Conflicting full tensors: {c_name}, not merged.")
            # TODO: Can decide what to store in the merged logs later.
            return c_name, None  # to pass my own c_name integrity checking
    else:  # shard tensors
        return c_name, TensorInfo.merge_tensor_info(log_shards)
        

def _merge_distr_logs(log_path: str, t_type: TType) -> None:
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
    
    # read logs
    log_collections: dict[str, list[TensorInfo]] = defaultdict(list)
    canonical_names: set[str] = set()  # just for checking
    for raw_log_file in raw_log_files:  # e.g.: rank0.pt
        raw_logs: dict[str, TensorInfo] = torch.load(raw_log_file, weights_only=False)
        for c_name, log in raw_logs.items():
            log_collections[c_name].append(log)
        canonical_names.update(raw_logs.keys())
    
    # merge logs in parallel
    # dict[str, list[TensorInfo]] -> dict[str, TensorInfo|None]
    # num_task = len(log_collections)
    # output_list = [None] * num_task
    # with ProcessPoolExecutor() as executor:
    #     futures = {
    #         executor.submit(_merge_logs, c_name, log_shards, t_type): i
    #         for (i, (c_name, log_shards)) in enumerate(log_collections.items())
    #     }
    #     pbar = tqdm(total=num_task, desc=f"Merging {t_type} logs", 
    #                 position=list(TType).index(t_type), leave=True)
    #     for future in as_completed(futures):
    #         i = futures[future]
    #         output_list[i] = future.result()
    #         pbar.update(1)
    #     pbar.close()

    # clean up and save
    # merged_logs: dict[str, TensorInfo | None] = dict(output_list)

    merged_logs = {}
    for c_name, log_shards in tqdm(
        log_collections.items(), 
        position=list(TType).index(t_type), 
        desc=f"Merging {t_type} logs"
    ):
        merged_logs[c_name] = _merge_logs(c_name, log_shards, t_type)[1]
    
    for canonical_name in canonical_names:
        assert canonical_name in merged_logs
    trim_word_embeddings(merged_logs)
    torch.save(merged_logs, merged_file_path)
    clog.info(f"Saved merged logs to {merged_file_path}")


def merge_distr_logs(
    log_path: str, 
    t_type: Optional[TType] = None,
    parallel: bool = False,
) -> None:
    """
    Merge the logs from different ranks into a single log file.
    Works for one specific tensor type.
    """
    clog_file = Path(log_path) / "merger.log"
    clog_file.touch()
    clog.add(clog_file)

    if not isinstance(t_type, TType) and t_type is not None:
        raise ValueError(f"Invalid type for `t_type`: {type(t_type)}")
    if t_type is not None:
        running_ttypes = [t_type]
    else:
        running_ttypes = list(TType)

    if not parallel:
        for t_type in running_ttypes:
            _merge_distr_logs(log_path, t_type)
    else:
        with ProcessPoolExecutor() as executor:
            futures = {
                executor.submit(_merge_distr_logs, log_path, t_type): t_type
                for t_type in running_ttypes
            }
            for future in as_completed(futures):
                future.result()
