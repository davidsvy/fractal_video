import datetime
import os
import re
import shutil
import cv2
import numpy as np
import time

from ..utils.data import dataset_stats, find_files
from ..utils.other import run_bash

from ..synthetic.io import save_video


url_help = 'https://stackoverflow.com/a/67550427'
id_gdrive = '17_uJgvxybtU-Pg0eh6PouYs0jSOOtpoq'
filename_zip = 'volleyball_.zip'

videos_train = [
    1, 3, 6, 7, 10, 13, 15, 16, 18, 22, 23, 31, 32,
    36, 38, 39, 40, 41, 42, 48, 50, 52, 53, 54,
]

videos_val = [
    0, 2, 8, 12, 17, 19, 24, 26, 27, 28, 30, 33, 46, 49, 51,
]

videos_test = [
    4, 5, 9, 11, 14, 20, 21, 25, 29, 34, 35, 37, 43, 44, 45, 47,
]


def resize(img, size):
    height, width = img.shape[:2]

    size += size % 2 == 1

    if height < width:
        width = int(width / height * size)
        width += width % 2 == 1
        dims_out = (width, size)
    else:
        height = int(height / width * size)
        height += height % 2 == 1
        dims_out = (size, height)

    img = cv2.resize(img, dims_out, interpolation=cv2.INTER_AREA)

    return img


def read_annotations(path):
    annotations = []

    with open(path, 'r') as file:
        for line in file:
            idx_center, label = re.split('\s+', line)[:2]
            idx_center = int(idx_center.split('.')[0])
            annotations.append((idx_center, label))

    return annotations


def extract_frames(idx_video, idx_center, n_window, dir_temp):
    dir = os.path.join(dir_temp, 'videos', f'{idx_video}', f'{idx_center}')
    idx_frames = list(range(idx_center - n_window, idx_center + n_window + 1))
    paths = [os.path.join(dir, f'{idx_frame}.jpg') for idx_frame in idx_frames]

    return paths


def frames_to_video(paths_frames, path_video, size=256, fps=25):
    frames = [resize(img=cv2.imread(path), size=size) for path in paths_frames]
    frames = np.stack(frames, axis=0)
    save_video(frames=frames, path=path_video, fps=fps, scale=False, lib='cv2')
    

def print_stats(root, ext):
    n_train = len(find_files(dir=os.path.join(root, 'train'), ext=ext))
    n_val = len(find_files(dir=os.path.join(root, 'val'), ext=ext))
    n_test = len(find_files(dir=os.path.join(root, 'test'), ext=ext))
    print(f'train -> {n_train} files')
    print(f'val -> {n_val} files')
    print(f'test -> {n_test} files')


def volleyball(root):
    time_start = time.time()

    path_zip = os.path.join(root, filename_zip)
    dir_temp = os.path.join(root, 'temp')
    n_videos, n_window = 55, 20
    resolution, fps = 256, 25
    ext = 'mp4'

    if not os.path.isfile(path_zip):
        msg = (
            f'Manually download {filename_zip} following {url_help}.\n'
            f'Run command:\n'
            f'curl -H "Authorization: Bearer [ACCESS_TOKEN]" https://www.googleapis.com/drive/v3/files/{id_gdrive}?alt=media -o {filename_zip} --http1.1\n'
            f'Move {filename_zip} to {root}.'
        )

        raise ValueError(msg)

    lut_split = {}
    for idx_video in videos_train:
        lut_split[idx_video] = 'train'
    for idx_video in videos_val:
        lut_split[idx_video] = 'val'
    for idx_video in videos_test:
        lut_split[idx_video] = 'test'

    if os.path.isdir(dir_temp):
        shutil.rmtree(dir_temp)

    for idx_video in range(n_videos):
        os.makedirs(dir_temp, exist_ok=True)

        run_bash(f'unzip {path_zip} videos/{idx_video}/* -d {dir_temp}')
        path_ann = os.path.join(
            dir_temp, 'videos', f'{idx_video}', 'annotations.txt')

        annotations = read_annotations(path_ann)

        for idx_center, label in annotations:
            paths_frames = extract_frames(
                idx_video=idx_video, idx_center=idx_center, n_window=n_window,
                dir_temp=dir_temp,
            )

            dir_tgt = os.path.join(root, lut_split[idx_video], label)
            path_tgt = os.path.join(dir_tgt, f'{idx_video}-{idx_center}.{ext}')

            os.makedirs(dir_tgt, exist_ok=True)
            frames_to_video(
                paths_frames=paths_frames, path_video=path_tgt, size=resolution, fps=fps)

        shutil.rmtree(dir_temp)

        so_far = time.time() - time_start
        time_avg = so_far / (idx_video + 1)
        eta = time_avg * (n_videos - idx_video - 1)
        so_far = datetime.timedelta(seconds=int(so_far))
        eta = datetime.timedelta(seconds=int(eta))

        print(
            f'Step [{idx_video + 1}/{n_videos}] -> SO-FAR: {so_far}, ETA: {eta}')

    dataset_stats(root=root, ext=ext)
    print(f'MANUALLY DELETE {path_zip}')


def volleyball_preprocessed(root):
    """
    train -> 2152 files
    val -> 1341 files
    test -> 1337 files
    """
    id, zip_file = '1--7js69pPQ7jh7g3_dcI5aSfCbuuoeK9', 'volleyball-512.zip'    

    os.makedirs(root, exist_ok=True)
    print('Downloading VOLLEYBALL...')
    run_bash(f'gdown {id}')
    print('Extracting VOLLEYBALL...')
    run_bash(f'unzip {zip_file} -d {root}')
    os.remove(zip_file)

    dataset_stats(root=root, ext='mp4')
