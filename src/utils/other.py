import numpy as np
import os
import random
import re
import subprocess


def run_bash(command):
    return subprocess.run(command, shell=True, capture_output=True, text=True)


def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


class Average_Meter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update_sum(self, val, n=1):
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count

    def update_avg(self, val, n=1):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def time_to_secs(timestamp):
    found = set()
    limit = {'s': 60, 'm': 60}
    coef = {unit: 60 ** step for step, unit in enumerate('smh')}
    secs = 0

    if not isinstance(timestamp, str):
        return 0

    splits = re.findall(r'\d+[smh]', timestamp)
    if len(''.join(splits)) != len(timestamp):
        return 0

    for split in splits:
        unit = split[-1]
        if unit in found:
            return 0
        found.add(unit)

        value = int(split[:-1])
        if unit != 'h' and value >= limit[unit]:
            return 0

        secs += (value * coef[unit])

    return secs


def size_to_str(size):
    units = ['KB', 'MB', 'GB']

    escape_flag = False
    for step in range(1, len(units)):
        new_size = size / 1024
        if new_size < 1:
            escape_flag = True
            break

        size = new_size

    str_ = f'{size:.1f}{units[step - escape_flag]}'

    return str_


def update_size_counter(counter, path):
    if isinstance(path, (list, tuple)):
        size = [os.path.getsize(p) for p in path]
        counter.update_sum(val=sum(size) / 1024, n=len(size))
    else:
        size = os.path.getsize(path) / 1024
        counter.update_sum(val=size, n=1)
        