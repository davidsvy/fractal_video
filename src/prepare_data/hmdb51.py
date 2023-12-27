import os
import pandas
import shutil

from ..utils.data import dataset_stats, find_files
from ..utils.other import run_bash


def hmdb51(root, split=1):
    """
    train -> 3570 files
    val -> 1530 files
    """
    assert split in [1, 2, 3]
    url_rar = 'http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/hmdb51_org.rar'
    url_split = 'http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/test_train_splits.rar'
    lut_split = {1: 'train', 2: 'val'}
    lut_csv = {'train': [[], []], 'val': [[], []]}

    path_rar = os.path.join(root, os.path.basename(url_rar))
    path_split = os.path.join(root, os.path.basename(url_split))

    dir_rar = os.path.join(root, 'rars')
    dir_split = os.path.join(root, 'splits')

    os.makedirs(dir_rar, exist_ok=True)
    os.makedirs(dir_split, exist_ok=True)

    print('\nDownloading HMDB51...')
    run_bash(f'wget {url_rar} -P {root} --no-check-certificate')
    run_bash(f'wget {url_split} -P {root} --no-check-certificate')

    print('Extracting HMDB51...')
    run_bash(f'unrar e {path_rar} {dir_rar}')
    os.remove(path_rar)
    run_bash(f'unrar e {path_split} {dir_split}')
    os.remove(path_split)

    all_rars = find_files(dir=dir_rar, ext='rar')

    for idx_class, rar_class in enumerate(all_rars):
        class_ = os.path.splitext(os.path.basename(rar_class))[0]
        txt_split = os.path.join(dir_split, f'{class_}_test_split{split}.txt')
        df_split = pandas.read_csv(txt_split, header=None, sep=' ')
        df_split = list(df_split[[0, 1]].to_records(index=False))

        dir_temp = os.path.join(root, 'temp')
        os.makedirs(dir_temp, exist_ok=True)
        run_bash(f'unrar e {rar_class} {dir_temp}')
        os.remove(rar_class)

        for filename, index in df_split:
            if not index in lut_split:
                continue

            path_src = os.path.join(dir_temp, filename)
            split_tgt = lut_split[index]

            dir_tgt = os.path.join(root, split_tgt, class_)
            path_tgt = os.path.join(dir_tgt, filename)

            os.makedirs(dir_tgt, exist_ok=True)
            shutil.move(path_src, path_tgt)

            lut_csv[split_tgt][0].append(path_tgt)
            lut_csv[split_tgt][1].append(idx_class)

        shutil.rmtree(dir_temp)

    for split_tgt in lut_csv:
        path_csv = os.path.join(root, f'{split_tgt}.csv')
        dict_csv = {
            'path': lut_csv[split_tgt][0], 'label': lut_csv[split_tgt][1]}
        df_csv = pandas.DataFrame(dict_csv)
        df_csv.to_csv(path_csv, sep=' ', header=False, index=False)

    shutil.rmtree(dir_rar)
    shutil.rmtree(dir_split)

    dataset_stats(root=root, ext='avi')
