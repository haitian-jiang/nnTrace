import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from typing import List


subnormals = {"bf16": 1.2e-38, "fp16": 6e-5}
epsilons = {"fp16": 9.8e-4, "bf16": 7.8e-3}


def load_fwd(single_file, multi_path):
    single = torch.load(single_file)
    m0 = torch.load(multi_path + '/rank0.pt')
    m4 = torch.load(multi_path + '/rank4.pt')
    m2 = torch.load(multi_path + '/rank2.pt')
    m6 = torch.load(multi_path + '/rank6.pt')
    distr = m0 + m4 + m2 + m6
    single_idx = {n: i for (i, (n, _)) in enumerate(single) if n.startswith('batch_')}  # batch_0.

    s = {}
    d = {}
    for (n, t) in distr:
        if n in single_idx:
            s[n] = single[single_idx[n]][1]
            d[n] = t

    return s, d, single, distr


def get_rel_err(oracle, specimen, dtype="bf16"):
    subnormal = subnormals[dtype]
    mask = (torch.abs(oracle) > subnormal) & (torch.abs(specimen) > subnormal)
    single = oracle[mask]
    distri = specimen[mask]
    rel_err = torch.abs(single - distri) / torch.abs(single)
    return rel_err.sort()[0]


def plot(dtype: str, errs: List[torch.Tensor], legend: List[str], ttype="Activation", ylim=None):
    epsilon = epsilons[dtype]
    plt.axhline(y=epsilon, color='r', linestyle='-', linewidth=0.5, label='ε')
    for err in errs:
        plt.plot(np.arange(len(err))/len(err), err.numpy())
    plt.yscale('log')
    plt.legend([f'ε({dtype})={epsilon:.0e}'] + legend)
    plt.xlabel('Percentage (Sorted by Relative Error)')
    plt.ylabel('Relative Error')
    plt.title(f"{dtype} | {ttype}")
    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(0.1))  # 每 1 个单位一个主刻度
    plt.gca().xaxis.set_minor_locator(ticker.MultipleLocator(0.05))  # 每 0.2 个单位一个次刻度
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    if ylim is not None:
        plt.ylim(ylim)
    # plt.show()