import os
import shutil

from ..utils.data import dataset_stats
from ..utils.other import run_bash


def correct_filename(filename):
    """Fixes small annotation error in UCF-101."""
    parts = filename.split('_')
    if parts[1] != 'HandStandPushups':
        return filename

    parts[1] = 'HandstandPushups'
    filename = '_'.join(parts)

    return filename


def move_files(paths, dir_src, dir_tgt):
    for path in paths:
        class_, filename = path.split('/')
        filename = correct_filename(filename)
        src_path = os.path.join(dir_src, filename)
        class_dir = os.path.join(dir_tgt, class_)
        tgt_path = os.path.join(class_dir, filename)

        os.makedirs(class_dir, exist_ok=True)
        shutil.move(src_path, tgt_path)


def ucf101(root, split=1):
    """
    train -> 9537 files
    val -> 3783 files
    """
    assert split in [1, 2, 3]
    url_data = 'http://storage.googleapis.com/thumos14_files/UCF101_videos.zip'
    url_label = 'https://www.crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip'

    path_data = os.path.join(root, os.path.basename(url_data))
    path_label = os.path.join(root, os.path.basename(url_label))

    dir_video = os.path.join(root, 'videos')
    dir_label = os.path.join(root, 'labels')
    dir_train = os.path.join(root, 'train')
    dir_val = os.path.join(root, 'val')

    txt_train = os.path.join(dir_label, f'trainlist0{split}.txt')
    txt_val = os.path.join(dir_label, f'testlist0{split}.txt')

    os.makedirs(root, exist_ok=True)
    print('\nDownloading UCF101...')
    run_bash(f'wget {url_label} -P {root} --no-check-certificate')
    run_bash(f'wget {url_data} -P {root} --no-check-certificate')

    print('Extracting UCF101...')
    run_bash(f'unzip {path_label} -d {root}')
    os.remove(path_label)
    os.rename(os.path.join(root, 'ucfTrainTestlist'), dir_label)

    run_bash(f'unzip {path_data} -d {root}')
    os.remove(path_data)
    os.rename(os.path.join(root, 'UCF101'), dir_video)

    with open(txt_train, 'r') as file:
        paths_train = file.readlines()
    paths_train = [path.split()[0] for path in paths_train]

    with open(txt_val, 'r') as file:
        paths_val = file.readlines()
    paths_val = [path.strip() for path in paths_val]

    print('Moving files to correct directories...')
    move_files(
        paths=paths_train, dir_src=dir_video, dir_tgt=dir_train)
    move_files(paths=paths_val, dir_src=dir_video, dir_tgt=dir_val)

    shutil.rmtree(dir_video)

    dataset_stats(root=root, ext='avi')
