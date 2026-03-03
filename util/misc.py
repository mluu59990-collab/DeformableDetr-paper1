# ------------------------------------------------------------------------
# Cleaned misc.py for modern PyTorch (2.x) and torchvision (0.15+)
# Compatible with Kaggle / Python 3.12
# ------------------------------------------------------------------------

import os
import subprocess
import time
import datetime
import pickle
from collections import defaultdict, deque
from typing import Optional, List

import torch
import torch.nn as nn
import torch.distributed as dist
from torch import Tensor
import torch.nn.functional as F


# ------------------------------------------------------------------------
# SmoothedValue
# ------------------------------------------------------------------------

class SmoothedValue(object):
    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device="cuda")
        dist.barrier()
        dist.all_reduce(t)
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        return torch.tensor(list(self.deque)).median().item()

    @property
    def avg(self):
        return torch.tensor(list(self.deque), dtype=torch.float32).mean().item()

    @property
    def global_avg(self):
        return self.total / max(self.count, 1)

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value
        )


# ------------------------------------------------------------------------
# Distributed helpers
# ------------------------------------------------------------------------

def is_dist_avail_and_initialized():
    return dist.is_available() and dist.is_initialized()


def get_world_size():
    return dist.get_world_size() if is_dist_avail_and_initialized() else 1


def get_rank():
    return dist.get_rank() if is_dist_avail_and_initialized() else 0


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


# ------------------------------------------------------------------------
# NestedTensor
# ------------------------------------------------------------------------

class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device, non_blocking=False):
        cast_tensor = self.tensors.to(device, non_blocking=non_blocking)
        cast_mask = self.mask.to(device, non_blocking=non_blocking) if self.mask is not None else None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)


def nested_tensor_from_tensor_list(tensor_list: List[Tensor]):
    if tensor_list[0].ndim != 3:
        raise ValueError("Only 3D tensors supported")

    max_size = [max(s) for s in zip(*[img.shape for img in tensor_list])]
    batch_shape = [len(tensor_list)] + max_size
    b, c, h, w = batch_shape

    dtype = tensor_list[0].dtype
    device = tensor_list[0].device

    tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
    mask = torch.ones((b, h, w), dtype=torch.bool, device=device)

    for img, pad_img, m in zip(tensor_list, tensor, mask):
        pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
        m[: img.shape[1], : img.shape[2]] = False

    return NestedTensor(tensor, mask)


# ------------------------------------------------------------------------
# Interpolate (clean version)
# ------------------------------------------------------------------------

def interpolate(input, size=None, scale_factor=None, mode="nearest", align_corners=None):
    return F.interpolate(
        input,
        size=size,
        scale_factor=scale_factor,
        mode=mode,
        align_corners=align_corners,
    )


# ------------------------------------------------------------------------
# Accuracy
# ------------------------------------------------------------------------

@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]

    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


# ------------------------------------------------------------------------
# Gradient norm
# ------------------------------------------------------------------------

def get_total_grad_norm(parameters, norm_type=2):
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)

    if len(parameters) == 0:
        return torch.tensor(0.)

    device = parameters[0].grad.device
    total_norm = torch.norm(
        torch.stack([
            torch.norm(p.grad.detach(), norm_type).to(device)
            for p in parameters
        ]),
        norm_type
    )
    return total_norm


# ------------------------------------------------------------------------
# Inverse sigmoid
# ------------------------------------------------------------------------

def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)