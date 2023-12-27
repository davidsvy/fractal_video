import os
import pandas
import shutil

from ..utils.data import dataset_stats, find_files
from ..utils.other import run_bash


def scenes(root):
    """https://vision.eecs.yorku.ca/research/dynamic-scenes/
    train -> 120 files
    val -> 1080  files
    """
    url_data = 'https://vision.eecs.yorku.ca/WebShare/YUP++.zip'
    path_zip = os.path.join(root, os.path.basename(url_data))
    dir_temp = os.path.join(root, 'temp')
    dir_train = os.path.join(root, 'train')
    dir_val = os.path.join(root, 'val')
    path_split = os.path.join(dir_temp, '10_90_randsplit_1.txt')

    os.makedirs(dir_temp, exist_ok=True)

    print('\nDownloading DYNAMIC SCENES...')
    run_bash(f'wget {url_data} -P {root} --no-check-certificate')

    print('Extracting DYNAMIC_SCENES...')
    run_bash(f'unzip {path_zip} -d {dir_temp}')
    os.remove(path_zip)

    df = pandas.read_csv(path_split, header=None, delimiter=' ')
    ext = 'mp4'

    for filename, split in zip(df[0].tolist(), df[1].tolist()):
        label = filename.split('_')[0]
        motion, _split = split.split('_')

        _motion = 'camera_stationary' if motion == 'static' else 'camera_moving'
        dir_split = dir_train if _split == 'train' else dir_val

        path_src = os.path.join(dir_temp, _motion, label, f'{filename}.{ext}')
        dir_tgt = os.path.join(dir_split, label)
        path_tgt = os.path.join(dir_tgt, f'{filename}.{ext}')

        os.makedirs(dir_tgt, exist_ok=True)
        shutil.move(path_src, path_tgt)

    shutil.rmtree(dir_temp)

    dataset_stats(root=root, ext=ext)
