# Copyright (c) Alibaba, Inc. and its affiliates.
import asyncio
import datetime as dt
import os
import re
import socket
import sys
import time
from asyncio.subprocess import PIPE, STDOUT
from typing import (Any, Callable, List, Mapping, Optional, Sequence, Tuple,
                    Type, TypeVar)

import numpy as np
import torch.distributed as dist
from transformers import HfArgumentParser

from .logger import get_logger
from .np_utils import stat_array
from .torch_utils import broadcast_string, is_dist

logger = get_logger()


def check_json_format(obj: Any) -> Any:
    if obj is None or isinstance(
            obj, (int, float, str, complex)):  # bool is a subclass of int
        return obj

    if isinstance(obj, Sequence):
        res = []
        for x in obj:
            res.append(check_json_format(x))
    elif isinstance(obj, Mapping):
        res = {}
        for k, v in obj.items():
            res[k] = check_json_format(v)
    else:
        res = repr(obj)  # e.g. function
    return res


def _get_version(work_dir: str) -> int:
    if os.path.isdir(work_dir):
        fnames = os.listdir(work_dir)
    else:
        fnames = []
    v_list = [-1]
    for fname in fnames:
        m = re.match(r'v(\d+)', fname)
        if m is None:
            continue
        v = m.group(1)
        v_list.append(int(v))
    return max(v_list) + 1


def add_version_to_work_dir(work_dir: str) -> str:
    """add version"""
    version = _get_version(work_dir)
    time = dt.datetime.now().strftime('%Y%m%d-%H%M%S')
    sub_folder = f'v{version}-{time}'
    if dist.is_initialized() and is_dist():
        sub_folder = broadcast_string(sub_folder)
    work_dir = os.path.join(work_dir, sub_folder)
    return work_dir


_T = TypeVar('_T')


def parse_args(class_type: Type[_T],
               argv: Optional[List[str]] = None) -> Tuple[_T, List[str]]:
    parser = HfArgumentParser([class_type])
    args, remaining_args = parser.parse_args_into_dataclasses(
        argv, return_remaining_strings=True)
    return args, remaining_args


def lower_bound(lo: int, hi: int, cond: Callable[[int], bool]) -> int:
    # The lower bound satisfying the condition "cond".
    while lo < hi:
        mid = (lo + hi) >> 1
        if cond(mid):
            hi = mid
        else:
            lo = mid + 1
    return lo


def upper_bound(lo: int, hi: int, cond: Callable[[int], bool]) -> int:
    # The upper bound satisfying the condition "cond".
    while lo < hi:
        mid = (lo + hi + 1) >> 1  # lo + (hi-lo+1)>>1
        if cond(mid):
            lo = mid
        else:
            hi = mid - 1
    return lo


def test_time(func: Callable[[], _T],
              number: int = 1,
              warmup: int = 0,
              timer: Optional[Callable[[], float]] = None) -> _T:
    # timer: e.g. time_synchronize
    timer = timer if timer is not None else time.perf_counter
    #
    ts = []
    res = None
    # warmup
    for _ in range(warmup):
        res = func()
    #
    for _ in range(number):
        t1 = timer()
        res = func()
        t2 = timer()
        ts.append(t2 - t1)
    #
    ts = np.array(ts)
    _, stat_str = stat_array(ts)
    # print
    logger.info(f'time[number={number}]: {stat_str}')
    return res


def read_multi_line() -> str:
    res = []
    prompt = '<<<[M] '
    while True:
        text = input(prompt) + '\n'
        prompt = ''
        res.append(text)
        if text.endswith('#\n'):
            res[-1] = text[:-2]
            break
    return ''.join(res)


async def run_and_get_log(*args, timeout=None):
    process = await asyncio.create_subprocess_exec(
        *args, stdout=PIPE, stderr=STDOUT)
    lines = []
    while True:
        try:
            line = await asyncio.wait_for(process.stdout.readline(), timeout)
        except asyncio.TimeoutError:
            break
        else:
            if not line:
                break
            else:
                lines.append(str(line))
    return process, lines


def run_command_in_subprocess(*args, timeout):
    if sys.platform == 'win32':
        loop = asyncio.ProactorEventLoop()
        asyncio.set_event_loop(loop)
    else:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    process, lines = loop.run_until_complete(
        run_and_get_log(*args, timeout=timeout))
    return (loop, process), lines


def close_loop(handler):
    loop, process = handler
    process.kill()
    loop.close()


def find_free_port() -> str:
    # Copied from https://github.com/facebookresearch/detectron2/blob/main/detectron2/engine/launch.py # noqa: E501
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(('', 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return port
