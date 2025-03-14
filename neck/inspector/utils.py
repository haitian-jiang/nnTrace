import os
import re
import torch
import numpy as np
from collections import defaultdict
from pathlib import Path
from neck.structs import TType
from loguru import logger as clog
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


def _err_fn(data):
    torch.set_num_threads(1)
    name, specimen, oracle = data
    rerr = tensor_rel_err(specimen, oracle)
    return name, rerr


def get_err_raw(
        specimen_path, 
        oracle_path, 
        t_type: TType,
    ) -> dict[str, float]:
    """
    Retruns: dict[c_name, rel_err]
    """
    specimen_dict = torch.load(specimen_path, weights_only=False)
    oracle_dict = torch.load(oracle_path, weights_only=False)
    assert specimen_dict.keys() == oracle_dict.keys(), "Keys mismatch"

    input_list = []
    for name in specimen_dict.keys():
        specimen = specimen_dict[name].tensor
        oracle = oracle_dict[name].tensor
        if specimen.shape != oracle.shape:
            assert "word_embeddings" in name
            vocab = min(specimen.shape[0], oracle.shape[0])
            specimen = specimen[:vocab]
            oracle = oracle[:vocab]
        input_list.append((name, specimen, oracle))
    
    ncpu = os.cpu_count()
    clog.info(f"Start calculating errors for {t_type}!")
    with concurrent.futures.ProcessPoolExecutor(max_workers=ncpu) as executor:
        results = list(executor.map(_err_fn, input_list))
    err_dict = dict(results)
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

    for t_type in TType:
        if (output_dir / f"{t_type}.pt").exists():
            continue
        err_dict = get_err_raw(
            specimen_run_dir / f"{t_type}.pt",
            oracle_run_dir / f"{t_type}.pt",
            t_type,
        )
        torch.save(err_dict, output_dir / f"{t_type}.pt")
        with open(output_dir / f"{t_type}.log", "w") as f:
            for name, err in err_dict.items():
                print(f"{name:<90} {err:.4e} {err/2**-7:10.1f}", file=f)
        clog.info(f"Finished comparing and storing {t_type}! ")

__all__ = ["get_err_raw", "get_err_array", "store_err_result"]
