# ------------------------------------------------------------------------
# Deformable DETR - Fully compatible misc.py for PyTorch 2.x
# Keeps full original functionality
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
import torchvision


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
        device = "cuda" if torch.cuda.is_available() else "cpu"
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device=device)
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
# Distributed utilities
# ------------------------------------------------------------------------

def is_dist_avail_and_initialized():
    return dist.is_available() and dist.is_initialized()


def get_world_size():
    return dist.get_world_size() if is_dist_avail_and_initialized() else 1


def get_rank():
    return dist.get_rank() if is_dist_avail_and_initialized() else 0


def get_local_size():
    if not is_dist_avail_and_initialized():
        return 1
    return int(os.environ.get("LOCAL_SIZE", 1))


def get_local_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return int(os.environ.get("LOCAL_RANK", 0))


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


# ------------------------------------------------------------------------
# All gather
# ------------------------------------------------------------------------

def all_gather(data):
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to(device)

    local_size = torch.tensor([tensor.numel()], device=device)
    size_list = [torch.tensor([0], device=device) for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    tensor_list = [
        torch.empty((max_size,), dtype=torch.uint8, device=device)
        for _ in size_list
    ]

    if tensor.numel() != max_size:
        padding = torch.empty((max_size - tensor.numel(),),
                              dtype=torch.uint8, device=device)
        tensor = torch.cat((tensor, padding), dim=0)

    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list


# ------------------------------------------------------------------------
# Reduce dict
# ------------------------------------------------------------------------

def reduce_dict(input_dict, average=True):
    world_size = get_world_size()
    if world_size < 2:
        return input_dict

    with torch.no_grad():
        names = []
        values = []
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])

        values = torch.stack(values, dim=0)
        dist.all_reduce(values)

        if average:
            values /= world_size

        return {k: v for k, v in zip(names, values)}


# ------------------------------------------------------------------------
# NestedTensor
# ------------------------------------------------------------------------

class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device, non_blocking=False):
        tensors = self.tensors.to(device, non_blocking=non_blocking)
        mask = self.mask.to(device, non_blocking=non_blocking) if self.mask is not None else None
        return NestedTensor(tensors, mask)

    def record_stream(self, *args, **kwargs):
        self.tensors.record_stream(*args, **kwargs)
        if self.mask is not None:
            self.mask.record_stream(*args, **kwargs)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)


def _max_by_axis(the_list):
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for i, item in enumerate(sublist):
            maxes[i] = max(maxes[i], item)
    return maxes


def nested_tensor_from_tensor_list(tensor_list: List[Tensor]):
    if tensor_list[0].ndim != 3:
        raise ValueError("Only 3D tensors supported")

    max_size = _max_by_axis([list(img.shape) for img in tensor_list])
    batch_shape = [len(tensor_list)] + max_size

    b, c, h, w = batch_shape
    dtype = tensor_list[0].dtype
    device = tensor_list[0].device

    tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
    mask = torch.ones((b, h, w), dtype=torch.bool, device=device)

    for img, pad_img, m in zip(tensor_list, tensor, mask):
        pad_img[:img.shape[0], :img.shape[1], :img.shape[2]].copy_(img)
        m[:img.shape[1], :img.shape[2]] = False

    return NestedTensor(tensor, mask)


# ------------------------------------------------------------------------
# Interpolate (modern safe)
# ------------------------------------------------------------------------

def interpolate(input, size=None, scale_factor=None, mode="nearest", align_corners=None):
    return F.interpolate(
        input,
        size=size,
        scale_factor=scale_factor,
        mode=mode,
        align_corners=align_corners
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
# Grad norm
# ------------------------------------------------------------------------

def get_total_grad_norm(parameters, norm_type=2):
    parameters = [p for p in parameters if p.grad is not None]
    if len(parameters) == 0:
        return torch.tensor(0.)

    norm_type = float(norm_type)
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
    x = x.clamp(0, 1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)
# ------------------------------------------------------------------------
# Init distributed mode (required by main.py)
# ------------------------------------------------------------------------

def init_distributed_mode(args):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    else:
        print("Not using distributed mode")
        args.distributed = False
        args.rank = 0
        args.world_size = 1
        args.gpu = 0
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = "nccl"
    print(
        f"| distributed init (rank {args.rank}): {args.dist_url}",
        flush=True,
    )

    dist.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )
    dist.barrier()
# ------------------------------------------------------------------------
# Git SHA
# ------------------------------------------------------------------------

def get_sha():
    try:
        cwd = os.path.dirname(os.path.abspath(__file__))
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd
        ).decode("ascii").strip()
        return sha
    except Exception:
        return "N/A"
# ------------------------------------------------------------------------
# Collate function (for DataLoader)
# ------------------------------------------------------------------------

def collate_fn(batch):
    batch = list(zip(*batch))
    batch[0] = nested_tensor_from_tensor_list(batch[0])
    return tuple(batch)
# ------------------------------------------------------------------------
# Metric Logger
# ------------------------------------------------------------------------

class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(f"{type(self).__name__} has no attribute {attr}")

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(f"{name}: {str(meter)}")
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''

        start_time = time.time()
        end = time.time()

        for obj in iterable:
            yield obj
            i += 1
            if i % print_freq == 0 or i == len(iterable):
                eta_seconds = (time.time() - start_time) / i * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                print(
                    f"{header} [{i}/{len(iterable)}]  eta: {eta_string}  {str(self)}"
                )
            end = time.time()