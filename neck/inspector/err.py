import torch


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


__all__ = ["tensor_rel_err"]
