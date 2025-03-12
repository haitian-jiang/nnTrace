import re
import torch
import numpy as np
from collections import defaultdict
from typing import Optional
from pathlib import Path
from neck.structs import TType
from loguru import logger as clog
from tqdm.auto import tqdm
import concurrent.futures
from .err import tensor_rel_err


import re
def parse_name(name):
    pattern = r"batch\.(\d+)\|.*?layers\.(\d+)[\|\.](\S+)"
    m = re.match(pattern, name)
    if m:
        batch_num = m.group(1)
        layer_num = m.group(2)
        activation_name = m.group(3)
        return batch_num, layer_num, activation_name
    else:
        return None, None, None





def _robust_tensor_rel_err(name, specimen, oracle):
    if specimen.shape != oracle.shape:
        assert "word_embeddings" in name
        vocab = min(specimen.shape[0], oracle.shape[0])
        specimen = specimen[:vocab]
        oracle = oracle[:vocab]
    return tensor_rel_err(specimen, oracle)


def get_err_raw(
        specimen_path, 
        oracle_path, 
        show_progress: bool = False,
        t_type: Optional[TType] = None,
    ) -> dict[str, float]:
    """
    Retruns: dict[c_name, rel_err]
    """
    specimen_dict = torch.load(specimen_path, weights_only=False)
    oracle_dict = torch.load(oracle_path, weights_only=False)
    assert specimen_dict.keys() == oracle_dict.keys(), "Keys mismatch"

    show_progress = t_type is TType.activ
    if show_progress:
        pbar = tqdm(total=len(oracle_dict), position=0, 
                    desc=f"Calculating {t_type} error")

    err_dict = {}
    for name in specimen_dict.keys():
        specimen = specimen_dict[name].tensor
        oracle = oracle_dict[name].tensor
        if specimen.shape != oracle.shape:
            assert "word_embeddings" in name
            cut = min(specimen.shape[0], oracle.shape[0])
            specimen = specimen[:cut]
            oracle = oracle[:cut]
        err = tensor_rel_err(specimen, oracle)
        err_dict[name] = err
        if show_progress:
            pbar.update(1)
    if show_progress:
        pbar.close()

    return err_dict


def get_err_array(err_dict, batch_num, layer_num) -> dict[str, np.ndarray]:
    err_array_dict = defaultdict(lambda: np.zeros((batch_num, layer_num)))
    for name, err in err_dict.items():
        batch, layer, activation_name = parse_name(name)
        if batch is None:
            continue
        err_array_dict[activation_name][int(batch), int(layer)] = err
    return err_array_dict


def store_err_result(
        oracle_run_dir: str, 
        specimen_run_dir: str, 
        output_dir: str,
    ):
    oracle_run_dir = Path(oracle_run_dir)
    specimen_run_dir = Path(specimen_run_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {
            t_type: executor.submit(
                get_err_raw,
                specimen_run_dir / f"{t_type}.pt",
                oracle_run_dir / f"{t_type}.pt",
                t_type is TType.activ,
                t_type,
            )
            for t_type in TType
        }
        results = {
            t_type: future.result()
            for t_type, future in futures.items()
        }

    for t_type in TType:
        torch.save(results[t_type], output_dir / f"{t_type}.pt")
    
    clog.info("Finished comparing and storing! ")
    
    for t_type in TType:
        with open(output_dir / f"{t_type}.log", "w") as f:
            for name, err in results[t_type].items():
                print(f"{name:<90} {err:.4e} {err/2**-7:10.1f}", file=f)


__all__ = ["get_err_raw", "get_err_array", "store_err_result"]
