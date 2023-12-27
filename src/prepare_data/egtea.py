import os
import re
import shutil

from ..utils.data import dataset_stats
from ..utils.other import run_bash


def move_files(path_txt_split, lut_label, dir_data, dir_split):
    with open(path_txt_split, 'r') as file:
        for line in file:
            filename, label = line.split(' ')[:2]
            label = lut_label[label]
            prefix = '-'.join(filename.split('-')[:3])

            path_src = os.path.join(dir_data, prefix, f'{filename}.mp4')
            dir_tgt = os.path.join(dir_split, label)
            path_tgt = os.path.join(dir_tgt, f'{filename}.mp4')

            os.makedirs(dir_tgt, exist_ok=True)
            shutil.move(path_src, path_tgt)


def egtea(root):
    """https://cbs.ic.gatech.edu/fpv/
    train -> 8299 files
    val -> 2022 files
    """
    url_data = 'https://www.dropbox.com/s/udynz2u62wpdva6/video_clips.tar?dl=1'
    url_ann = 'https://www.dropbox.com/s/ksro6eqa6v59859/action_annotation.zip?dl=1'

    path_data = os.path.join(root, os.path.basename(url_data))
    path_ann = os.path.join(root, os.path.basename(url_ann))

    dir_data = os.path.join(root, 'cropped_clips')
    dir_ann = os.path.join(root, 'ann')
    dir_train = os.path.join(root, 'train')
    dir_val = os.path.join(root, 'val')

    os.makedirs(dir_data, exist_ok=True)
    os.makedirs(dir_ann, exist_ok=True)

    print('\nDownloading EGTEA...')
    run_bash(f'wget {url_ann} -P {root} --no-check-certificate')
    assert os.path.isfile(path_ann)
    run_bash(f'wget {url_data} -P {root} --no-check-certificate')
    assert os.path.isfile(path_data)

    print('Extracting EGTEA...')
    run_bash(f'unzip {path_ann} -d {dir_ann}')
    os.remove(path_ann)
    run_bash(f'tar -xf {path_data} -C {root}')
    os.remove(path_data)

    path_label = os.path.join(dir_ann, 'action_idx.txt')
    lut_label = {}

    with open(path_label, 'r') as file:
        for line in file:
            line = line.strip().split(' ')
            idx = line[-1]
            name = '-'.join(line[:-1])
            name = re.sub(r'\W', '-', name)
            lut_label[idx] = name

    path_split_train = os.path.join(dir_ann, 'train_split1.txt')
    path_split_val = os.path.join(dir_ann, 'test_split1.txt')

    move_files(
        path_txt_split=path_split_train, lut_label=lut_label,
        dir_data=dir_data, dir_split=dir_train,
    )

    move_files(
        path_txt_split=path_split_val, lut_label=lut_label,
        dir_data=dir_data, dir_split=dir_val,
    )

    shutil.rmtree(dir_ann)
    shutil.rmtree(dir_data)

    dataset_stats(root=root, ext='mp4')


def egtea_preprocessed(root):
    """
    train -> 8299 files
    val -> 2022 files
    """
    id, zip_file = '1pYBAE0azrM7EqFtcm44MpyRalBAyyq44', 'egtea-256.zip'

    os.makedirs(root, exist_ok=True)
    print('Downloading EGTEA...')
    run_bash(f'gdown {id}')
    print('Extracting EGTEA...')
    run_bash(f'unzip {zip_file} -d {root}')
    os.remove(zip_file)

    dataset_stats(root=root, ext='mp4')
