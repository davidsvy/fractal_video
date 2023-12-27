import itertools
import os
import pprint
import shutil

from .gdrive_ids import lut_id, lut_info
from ..utils.data import find_classes, find_files, split_stats
from ..utils.other import run_bash


def parse_gdrive_filename(msg):
    if msg.returncode != 0:
        print(
            f'Command "{msg.args}" has following error:\n{msg.stdout}\n{msg.stderr}')

    filaname = msg.stderr.split('\n')[3][4:]
    filaname = os.path.basename(filaname)

    return filaname


def gdrive(dataset, root):
    assert dataset in lut_id and dataset in lut_info

    ids = lut_id[dataset]
    *_, extension = lut_info[dataset]

    print(f'\nDownloading & extracting {dataset} dataset...')
    print('#' * 50)

    for idx, id in enumerate(ids, 1):
        dir_tgt = os.path.join(root, f'{idx}')
        os.makedirs(dir_tgt, exist_ok=True)

        msg_gdrive = run_bash(f'gdown {id}')
        path_src = parse_gdrive_filename(msg_gdrive)

        run_bash(f'unzip {path_src} -d {dir_tgt}')
        os.remove(path_src)

        n_files = len(find_files(dir=dir_tgt, ext=extension))
        print(f'{path_src} ({idx}/{len(ids)}) -> {n_files} files')

    all_files = find_files(dir=root, ext=extension)

    print('#' * 50)
    print(f'Downloaded {len(all_files)} files')

    return all_files


def gdrive_supervised(dataset, root):
    assert dataset in lut_id and dataset in lut_info
    n_train, n_classes, n_val_class, ext = lut_info[dataset]

    paths_src = gdrive(dataset=dataset, root=root)

    print('#' * 50)
    print(f'Splitting {dataset} dataset into train & val set...')
    print('#' * 50)
    dir_train = os.path.join(root, 'train')
    dir_val = os.path.join(root, 'val')

    labels, _ = find_classes(paths_src)
    n_classes_full = labels.max() + 1
    if n_classes is None:
        n_classes = n_classes_full
    else:
        n_classes = min(n_classes, n_classes_full)

    n_val = n_val_class * n_classes
    #assert len(paths_src) >= n_train + n_val

    bins_class = [[] for _ in range(n_classes_full)]

    for path, label in zip(paths_src, labels):
        bins_class[label].append(path)

    bins_class = bins_class[:n_classes]

    assert n_val_class <= min(len(bin) for bin in bins_class)

    paths_src_sorted = []
    for group in itertools.zip_longest(*bins_class):
        paths_src_sorted += [path for path in group if path is not None]

    paths_src_val = paths_src_sorted[:n_val]
    paths_src_train = paths_src_sorted[n_val: n_val + n_train]

    for path_src in paths_src_train:
        path_rel = os.path.relpath(path_src, root)
        path_tgt = os.path.join(dir_train, path_rel)
        dir_tgt = os.path.dirname(path_tgt)

        os.makedirs(dir_tgt, exist_ok=True)
        shutil.move(path_src, path_tgt)

    for path_src in paths_src_val:
        path_rel = os.path.relpath(path_src, root)
        path_tgt = os.path.join(dir_val, path_rel)
        dir_tgt = os.path.dirname(path_tgt)

        os.makedirs(dir_tgt, exist_ok=True)
        shutil.move(path_src, path_tgt)

    all_subdir = os.listdir(root)
    all_subdir = [os.path.join(root, path) for path in all_subdir]
    all_subdir = [path for path in all_subdir if os.path.isdir(path)]

    for path in all_subdir:
        if not path in [dir_train, dir_val]:
            shutil.rmtree(path)

    # Sanity check
    stats_train = split_stats(dir=dir_train, ext=ext)
    print('-Train stats:')
    pprint.pprint(stats_train)

    stats_val = split_stats(dir=dir_val, ext=ext)
    print('-Val stats:')
    pprint.pprint(stats_val)


def gdrive_unsupervised(dataset, root):
    assert dataset in lut_id and dataset in lut_info
    n_train, *_, ext = lut_info[dataset]

    paths = gdrive(dataset=dataset, root=root)

    if len(paths) > n_train:
        for path in paths[n_train:]:
            os.remove(path)

    paths = find_files(dir=root, ext=ext)

    print('#' * 50)
    print(f'Kept {len(paths)} files')
