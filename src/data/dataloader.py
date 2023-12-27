import os
import torch

from .dataset import (
    collate_labeled_val,
    Dataset_Contrastive,
    Dataset_Labeled_Train,
    Dataset_Labeled_Val,
)

from ..transform.build import transform_inner
from ..transform.compose import transform_inner_train
from ..utils.data import find_classes, find_files
from ..utils.torch import LUT_Simple


##########################################################################
##########################################################################
# LABELED
##########################################################################
##########################################################################


def loader_labeled(config):
    dataset_train, dataset_val, lut_val = dataset_labeled(config)

    loader_train = torch.utils.data.DataLoader(
        dataset_train,
        shuffle=True,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.TORCH_N_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    loader_val = torch.utils.data.DataLoader(
        dataset_val,
        shuffle=False,
        batch_size=config.DATA.BATCH_SIZE_VAL,
        num_workers=config.DATA.TORCH_N_WORKERS,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_labeled_val,
    )

    return loader_train, loader_val, lut_val


def dataset_labeled(config):
    dataset_name = config.DATA.DATASET
    extension = config.DATA.EXTENSION
    root = os.path.join(config.DATA.ROOT, dataset_name)

    dir_train = os.path.join(root, 'train')
    dir_val = os.path.join(root, 'val')

    paths_train = find_files(dir=dir_train, ext=extension)
    assert paths_train, f'No files found at {dir_train}'
    labels_train, label_dict = find_classes(paths_train)

    paths_val = find_files(dir=dir_val, ext=extension)
    assert paths_val, f'No files found at {dir_val}'
    lut_val, _ = find_classes(paths=paths_val, label_dict=label_dict)
    lut_val = LUT_Simple(lut_val)
    labels_val = list(range(len(paths_val)))

    n_classes = labels_train.max() + 1
    steps_per_epoch = len(paths_train) // config.DATA.BATCH_SIZE
    config.defrost()
    config.MODEL.N_CLASSES = int(n_classes)
    config.STEPS_PER_EPOCH = steps_per_epoch
    config.freeze()

    transform_train = transform_inner(is_train=True, config=config)
    transform_val = transform_inner(is_train=False, config=config)

    print(f'\nDataset "{dataset_name}" stored in {root}:')
    print(f'\t#classes: {n_classes}')
    print(f'\t#train files: {len(paths_train)}')
    print(f'\t#val files: {len(paths_val)}')

    dataset_train = Dataset_Labeled_Train(
        paths=paths_train,
        labels=labels_train,
        clip_length=config.DATA.CLIP_LENGTH,
        stride=config.DATA.STRIDE,
        transform=transform_train,
        n_threads=config.DATA.DECORD_N_THREADS,
    )

    
    dataset_val = Dataset_Labeled_Val(
        paths=paths_val,
        labels=labels_val,
        clip_length=config.DATA.CLIP_LENGTH,
        min_step=config.DATA.VAL_MIN_STEPS,
        max_segs=config.DATA.VAL_MAX_SEGS,
        stride=config.DATA.STRIDE,
        transform=transform_val,
        n_threads=config.DATA.DECORD_N_THREADS,
    )

    return dataset_train, dataset_val, lut_val


def loader_test(config):
    dataset_name = config.DATA.DATASET
    extension = config.DATA.EXTENSION
    root = os.path.join(config.DATA.ROOT, dataset_name)

    dir_train = os.path.join(root, 'train')
    dir_test = os.path.join(root, 'val')

    paths_train = find_files(dir=dir_train, ext=extension)
    assert paths_train, f'No files found at {dir_train}'
    _, label_dict = find_classes(paths_train)

    paths_test = find_files(dir=dir_test, ext=extension)
    assert paths_test, f'No files found at {dir_test}'
    lut_test, _ = find_classes(paths=paths_test, label_dict=label_dict)
    n_classes = lut_test.max() + 1
    lut_test = LUT_Simple(lut_test)
    labels_test = list(range(len(paths_test)))

    config.defrost()
    config.MODEL.N_CLASSES = int(n_classes)
    config.freeze()

    transform = transform_inner(is_train=False, config=config)

    print(f'\nDataset "{dataset_name}" stored in {dir_test}:')
    print(f'\t#classes: {n_classes}')
    print(f'\t#files: {len(paths_test)}')

    dataset = Dataset_Labeled_Val(
        paths=paths_test,
        labels=labels_test,
        clip_length=config.DATA.CLIP_LENGTH,
        min_step=config.DATA.VAL_MIN_STEPS,
        max_segs=config.DATA.VAL_MAX_SEGS,
        stride=config.DATA.STRIDE,
        transform=transform,
        n_threads=config.DATA.DECORD_N_THREADS,
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        shuffle=False,
        batch_size=config.DATA.BATCH_SIZE_VAL,
        num_workers=config.DATA.TORCH_N_WORKERS,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_labeled_val,
    )

    return loader, lut_test


##########################################################################
##########################################################################
# UNLABELED/CONTRASTIVE
##########################################################################
##########################################################################


def loader_contrastive(config):
    dataset_name = config.DATA.DATASET
    extension = config.DATA.EXTENSION
    root = os.path.join(config.DATA.ROOT, dataset_name)

    transform = transform_inner_train(
        crop_size=config.DATA.IMG_SIZE,
        min_scale=config.AUG.MIN_SCALE,
        interp=config.AUG.INTERP,
    )

    paths = find_files(dir=root, ext=extension)
    assert len(paths), f'No files found at {root}'

    steps_per_epoch = len(paths) // config.DATA.BATCH_SIZE
    config.defrost()
    config.STEPS_PER_EPOCH = steps_per_epoch
    config.freeze()

    labels = list(range(len(paths)))

    dataset = Dataset_Contrastive(
        paths=paths,
        labels=labels,
        clip_length=config.DATA.CLIP_LENGTH,
        stride=config.DATA.STRIDE,
        transform=transform,
        n_threads=config.DATA.DECORD_N_THREADS,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=True,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.TORCH_N_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    return dataloader


##########################################################################
##########################################################################
# OTHER
##########################################################################
##########################################################################


def unpack_contrastive(batch, device):
    return batch[0].to(device), batch[1].to(device)


def unpack_labeled(batch, device):
    return batch[0].to(device), batch[1].to(device)
