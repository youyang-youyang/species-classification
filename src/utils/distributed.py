"""
分布式训练工具

支持:
  - 自动检测 GPU 数量
  - 单卡/多卡模式切换
  - DDP 初始化与清理
"""

import os
import torch
import torch.distributed as dist


def setup_distributed():
    """
    初始化分布式训练环境

    Returns:
        rank: 当前进程的 rank
        world_size: 总进程数
        device: 当前进程使用的设备
    """
    if 'WORLD_SIZE' in os.environ:
        # torchrun 启动的多卡模式
        world_size = int(os.environ['WORLD_SIZE'])
        rank = int(os.environ['LOCAL_RANK'])
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(rank)
        device = torch.device(f'cuda:{rank}')
        if rank == 0:
            print(f"[DDP] World size: {world_size}, Backend: nccl")
    else:
        # 单卡模式
        rank = 0
        world_size = 1
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[Single GPU] Device: {device}")

    return rank, world_size, device


def cleanup_distributed():
    """清理分布式环境"""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    """是否为主进程 (rank 0)"""
    if dist.is_initialized():
        return dist.get_rank() == 0
    return True


def reduce_mean(tensor: torch.Tensor, world_size: int) -> torch.Tensor:
    """跨进程取平均"""
    if world_size <= 1:
        return tensor
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt


def print_rank0(*args, **kwargs):
    """只在 rank 0 打印"""
    if is_main_process():
        print(*args, **kwargs)
