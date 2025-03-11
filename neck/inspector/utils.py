import re
import torch
import numpy as np
from collections import defaultdict
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
    

def get_err_raw(specimen_path, oracle_path) -> dict[str, float]:
    specimen_dict = torch.load(specimen_path)
    oracle_dict = torch.load(oracle_path)
    assert specimen_dict.keys() == oracle_dict.keys(), "Keys mismatch"
    err_dict = {}
    for name in specimen_dict.keys():
        specimen = specimen_dict[name].tensor
        oracle = oracle_dict[name].tensor
        if oracle.shape != specimen.shape:
            assert "word_embedding" in name
            specimen = specimen.view(-1)[:oracle.numel()].view(oracle.shape)
        err = tensor_rel_err(specimen, oracle)
        err_dict[name] = err
    return err_dict


def get_err_array(err_dict, batch_num, layer_num) -> dict[str, np.ndarray]:
    err_array_dict = defaultdict(lambda: np.zeros((batch_num, layer_num)))
    for name, err in err_dict.items():
        batch, layer, activation_name = parse_name(name)
        if batch is None:
            continue
        err_array_dict[activation_name][int(batch), int(layer)] = err
    return err_array_dict
