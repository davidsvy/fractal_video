import json
import os
import shutil

from ..utils.data import dataset_stats
from ..utils.other import run_bash


def move_files(path_split, dir_src, dir_tgt, ext):
    with open(path_split, 'r') as file:
        lut = json.load(file)

        for item in lut:
            filename = f'{item["vid_name"]}.{ext}'
            path_src = os.path.join(dir_src, filename)
            label = str(item['label'])
            dir_label = os.path.join(dir_tgt, label)
            path_tgt = os.path.join(dir_label, filename)

            os.makedirs(dir_label, exist_ok=True)
            shutil.move(path_src, path_tgt)


def diving48(root):
    """
    train -> 15943 files
    val -> 2096 files
    """
    url_data = 'http://www.svcl.ucsd.edu/projects/resound/Diving48_rgb.tar.gz'
    url_split_train = 'http://www.svcl.ucsd.edu/projects/resound/Diving48_train.json'
    url_split_val = 'http://www.svcl.ucsd.edu/projects/resound/Diving48_test.json'

    path_data = os.path.join(root, os.path.basename(url_data))
    path_split_train = os.path.join(root, os.path.basename(url_split_train))
    path_split_val = os.path.join(root, os.path.basename(url_split_val))

    dir_src = os.path.join(root, 'rgb')
    dir_train = os.path.join(root, 'train')
    dir_val = os.path.join(root, 'val')
    ext = 'mp4'

    os.makedirs(dir_train, exist_ok=True)
    os.makedirs(dir_val, exist_ok=True)

    print('\nDownloading DIVING48...')
    run_bash(f'wget {url_split_train} -P {root}')
    run_bash(f'wget {url_split_val} -P {root}')
    run_bash(f'wget {url_data} -P {root}')

    print('Extracting DIVING48...')
    run_bash(f'tar -xf {path_data} -C {root}')
    os.remove(path_data)

    move_files(
        path_split=path_split_train, dir_src=dir_src,
        dir_tgt=dir_train, ext=ext
    )

    move_files(
        path_split=path_split_val, dir_src=dir_src,
        dir_tgt=dir_val, ext=ext
    )

    shutil.rmtree(dir_src)
    os.remove(path_split_train)
    os.remove(path_split_val)

    dataset_stats(root=root, ext=ext)


def diving48_preprocessed(root):
    """
    train -> 15943 files
    val -> 2096 files
    """
    id, zip_file = '1685eDcMitF4-jNXAn7Y_rpYH_LHkdFk7', 'diving48-256.zip'

    os.makedirs(root, exist_ok=True)
    print('Downloading DIVING48...')
    run_bash(f'gdown {id}')
    print('Extracting DIVING48...')
    run_bash(f'unzip {zip_file} -d {root}')
    os.remove(zip_file)

    dataset_stats(root=root, ext='mp4')
