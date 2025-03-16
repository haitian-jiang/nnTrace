import re
import torch
import numpy as np
from collections import defaultdict
from pathlib import Path
from neck.structs import TType
from loguru import logger as clog
from typing import Optional
from concurrent.futures import ProcessPoolExecutor
from neck import cpp_ext as neck_backend


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


def tensor_rel_err(specimen, oracle, norm='fro'):
    specimen = specimen.to(torch.float64)
    oracle = oracle.to(torch.float64)
    mask = specimen.isnan() | oracle.isnan() | specimen.isinf() | oracle.isinf()
    specimen[mask] = 0
    oracle[mask] = 0
    if norm == 'fro':
        rerr = (specimen - oracle).norm() / oracle.norm()
    elif norm == '1':
        rerr = (specimen - oracle).abs().sum() / oracle.abs().sum()
    elif norm == 'inf':
        rerr = (specimen - oracle).abs().max() / oracle.abs().max()
    return rerr.item()


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
    num_task = len(input_list)
    output_list = [None] * num_task
    
    clog.info(f"Start calculating errors for {t_type}!")
    # list[tuple[str, Tensor, Tensor]] -> list[tuple[str, double]]
    output_list = neck_backend.parallel_rel_err(input_list)
    clog.info(f"Finished calculating errors for {t_type}!")

    return dict(output_list)


def _get_err_raw_and_store_result(
        specimen_path,
        oracle_path,
        output_dir,
        t_type: TType,
    ) -> bool:
    err_dict = get_err_raw(specimen_path, oracle_path, t_type)
    torch.save(err_dict, output_dir / f"{t_type}.pt")
    with open(output_dir / f"{t_type}.log", "w") as f:
        for name, err in err_dict.items():  # TODO: hard-coded 2**-7
            print(f"{name:<90} {err:.4e} {err/2**-7:10.1f}", file=f)
    clog.info(f"Finished storing error results for {t_type}!")
    return True  # success


def get_err_array(err_dict, batch_num, layer_num) -> dict[str, np.ndarray]:
    err_array_dict = defaultdict(lambda: np.zeros((batch_num, layer_num)))
    for name, err in err_dict.items():
        batch, layer, activation_name = parse_name(name)
        if batch is None:
            continue
        err_array_dict[activation_name][int(batch), int(layer)] = err
    return err_array_dict


def conduct_differential_test(
        oracle_run_dir: str, 
        specimen_run_dir: str, 
        output_dir: str,
        t_type: Optional[TType] = None,
    ):
    oracle_run_dir = Path(oracle_run_dir)
    specimen_run_dir = Path(specimen_run_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # form tasks
    if not isinstance(t_type, TType) and t_type is not None:
        raise ValueError(f"Invalid type for `t_type`: {type(t_type)}")
    if t_type is not None:
        running_ttypes = [t_type]
    else:
        running_ttypes = list(TType)

    # screen out the ones that are already calculated
    running_ttypes = [
        t_type 
        for t_type in running_ttypes 
        if not (output_dir / f"{t_type}.pt").exists()
    ]

    with ProcessPoolExecutor() as executor:
        futures = {
            t_type: executor.submit(
                _get_err_raw_and_store_result,
                specimen_run_dir / f"{t_type}.pt",
                oracle_run_dir / f"{t_type}.pt",
                output_dir,
                t_type,
            )
            for t_type in running_ttypes
        }
        results = {
            t_type: future.result()
            for t_type, future in futures.items()
        }
        assert all(results.values()), "Some tasks failed!"


__all__ = [
    "parse_name",
    "tensor_rel_err",
    "get_err_raw", 
    "get_err_array", 
    "conduct_differential_test",
]
