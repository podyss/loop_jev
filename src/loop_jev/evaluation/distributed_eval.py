"""Data-parallel evaluation with small CPU collectives."""

import os
from contextlib import contextmanager

import torch
import torch.distributed as dist


def eval_rank():
    return dist.get_rank() if dist.is_initialized() else 0


def eval_world_size():
    return dist.get_world_size() if dist.is_initialized() else 1


def eval_shard(rows):
    return rows[eval_rank()::eval_world_size()]


@contextmanager
def evaluation_device():
    device = torch.device('cuda', int(os.environ.get('LOCAL_RANK', 0)))
    torch.cuda.set_device(device)
    distributed = int(os.environ.get('WORLD_SIZE', 1)) > 1
    if distributed:
        dist.init_process_group('gloo')
    try:
        yield device
    finally:
        if distributed:
            dist.destroy_process_group()


def gather_rows(local, total):
    """Gather strided shards and restore their original sample order."""
    world = eval_world_size()
    if world == 1:
        return local
    cpu = local.cpu()
    width = (total + world - 1) // world
    padded = cpu.new_zeros((width, *cpu.shape[1:]))
    padded[:len(cpu)] = cpu
    gathered = [torch.empty_like(padded) for _ in range(world)]
    dist.all_gather(gathered, padded)
    return torch.stack(gathered, dim=1).flatten(0, 1)[:total].to(local.device)
