import os
import math
from contextlib import nullcontext

import torch
import numpy as np
import torch.distributed as dist
import torch.nn as nn
from torch.distributed import destroy_process_group, init_process_group

from model import Block, GPTConfig


use_tensor_parallel = int(os.environ.get("RANK", -1)) != -1
_TP_Ready = False
if use_tensor_parallel:
    try:
        from torch.distributed._tensor import DeviceMesh
        from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel, PairwiseParallel, parallelize_module
        from torch.distributed.tensor.parallel._utils import _create_1d_device_mesh
        _TP_Ready = True
    except ImportError as e:
        print(f"Tensor Parallel is not available: {e}")
    assert _TP_Ready, "Tensor Parallel is not available"

gradient_accumulation_steps = 4
batch_size = 12 
block_size = 1024
n_head = 16
n_embd = 768
dropout = 0.0
bias = False
learning_rate = 1e-4
max_iters = 600000
weight_decay = 1e-2
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
backend = "nccl"
device = "cuda"

if use_tensor_parallel:
    rank = int(os.environ.get("RANK"))
    world_size = int(os.environ.get("WORLD_SIZE"))
    local_rank = int(os.environ.get("LOCAL_RANK"))
    device = f"cuda:{local_rank}"
    init_process_group(backend=backend)
    torch.cuda.set_device(device)
    master_process = rank == 0
    seed_offset = rank
    if _TP_Ready:
        mesh = DeviceMesh(device_type="cuda", mesh=torch.arange(0, world_size))
    else:
        mesh = None
else:
    rank = 0
    device = "cuda"
    master_process = True
    seed_offset = 0
    mesh = None

def rank_print(x):
    if rank == 0:
        print(x)

rank_print(f"{mesh=}\n")

torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = "cuda" if "cuda" in device else "cpu" 

dtype = "bfloat16"  # 'float32', 'bfloat16' 
ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)


def get_batch():
    x = torch.rand(batch_size, block_size, n_embd, dtype=ptdtype, device=device)
    y = torch.rand(batch_size, block_size, n_embd, dtype=ptdtype, device=device)
    return x, y


iter_num = 0
best_val_loss = float("inf")


model_args = dict(n_head=n_head, n_embd=n_embd, block_size=block_size, bias=bias, dropout=dropout)
gpt_conf = GPTConfig(**model_args)
model = Block(gpt_conf, mesh=mesh)

if use_tensor_parallel:
    model = parallelize_module(
        module=model, 
        device_mesh=mesh,
        parallelize_plan={
            "attn.c_attn": ColwiseParallel(),
            "attn.c_proj": RowwiseParallel(),
            "mlp": PairwiseParallel(),
        },
        tp_mesh_dim=0,
    )
    rank_print("Initialized Tensor Parallelism")

model.to(device)
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == "float16"))
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(beta1, beta2), weight_decay=weight_decay)


if use_tensor_parallel:
    _tp_size = mesh.mesh.size(0)
    rank_print(f"{_tp_size=}")
else:
    _tp_size = 1

local_iter_num = 0

X, Y = get_batch()
while local_iter_num < max_iters:
    model.train()
    for micro_step in range(gradient_accumulation_steps):
        with ctx:
            logits, loss = model(X, Y)
        X, Y = get_batch()
        scaler.scale(loss).backward()
    # if grad_clip != 0.0:
    #     scaler.unscale_(optimizer)
    #     torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    iter_num += 1
    local_iter_num += 1
    if iter_num > max_iters:
        break

dist.barrier()
destroy_process_group()