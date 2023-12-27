import datetime
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from cfg._default import get_default_cfg
from src.data import loader_labeled, unpack_labeled
from src.encoder import build_encoder, init_sup
from src.ssl import init_ssl
from src.transform import build_mixup, transform_outer_train

from src.utils.args import parse_args_train
from src.utils.logger import create_logger
from src.utils.lr_scheduler import build_scheduler
from src.utils.optimizer import build_optimizer
from src.utils.other import Average_Meter, set_seed, time_to_secs

from src.utils.torch import (
    accuracy, 
    count_parameters, 
    get_device, 
    load_checkpoint,
    save_checkpoint,
    set_seed,
)


def main():
    config = get_default_cfg()
    args = parse_args_train()

    config.defrost()
    if args.cfg is not None:
        for _config in args.cfg:
            config.merge_from_file(_config)

    if args.device is not None:
        config.DEVICE = args.device

    if args.opts is not None:
        config.merge_from_list(args.opts)
        
    config.TRAIN.MODE = 'supervised'
        
    config.freeze()
    
    main_worker(config)


def main_worker(config):
    count_ckpt = 0
    if config.MODEL.RESUME:
        assert os.path.isfile(config.MODEL.RESUME)
        count_ckpt += 1
    if config.TRAIN.INIT_SSL:
        assert os.path.isfile(config.TRAIN.INIT_SSL)
        count_ckpt += 1
    if config.TRAIN.INIT_SUP:
        assert os.path.isfile(config.TRAIN.INIT_SUP)
        count_ckpt += 1

    assert count_ckpt <= 1, 'Provide only single source for checkpoint.'
    
    set_seed(config.SEED)

    config.defrost()

    config.TRAIN.TIME_LIMIT = time_to_secs(config.TRAIN.TIME_LIMIT)    
    config.TRAIN.LR_BASE = config.TRAIN.LR_BASE * config.DATA.BATCH_SIZE / 32
    config.TRAIN.LR_WARMUP = config.TRAIN.LR_WARMUP * config.DATA.BATCH_SIZE / 32
    config.TRAIN.LR_MIN = config.TRAIN.LR_MIN * config.DATA.BATCH_SIZE / 32

    config.freeze()

    os.makedirs(config.TRAIN.OUTPUT, exist_ok=True)
    logger = create_logger(dir=config.TRAIN.OUTPUT)
    device = get_device(gpu_id=config.DEVICE, log_fn=logger.info)

    path_config = os.path.join(config.TRAIN.OUTPUT, 'config.yaml')
    with open(path_config, 'w') as file:
        file.write(config.dump())

    logger.info(f'Full config saved to {path_config}')
    logger.info(config.dump())

    loader_train, loader_val, lut_val = loader_labeled(config)
    fn_unpack = unpack_labeled
    fn_metric = accuracy

    if lut_val is not None:
        lut_val = lut_val.to(device)

    transform = transform_outer_train(config)
    mixup = build_mixup(config)
    fn_loss = F.cross_entropy

    if config.TRAIN.INIT_SUP:
        logger.info(
            f'Initializing model from supervised checkpoint at {config.TRAIN.INIT_SUP}.')
        model = init_sup(config=config, logger=logger)

    elif config.TRAIN.INIT_SSL:
        logger.info(
            f'Initializing model from self-supervised checkpoint at {config.TRAIN.INIT_SSL}.')
        model = init_ssl(config=config, logger=logger)

    else:
        logger.info('Initializing model from scratch.')
        model = build_encoder(config=config, mlp_head=False)

    model = model.to(device)

    optimizer = build_optimizer(config=config, model=model)
    lr_scheduler = build_scheduler(
        config=config, optimizer=optimizer, steps_per_epoch=len(loader_train))
    scaler = torch.cuda.amp.GradScaler() if config.TRAIN.AMP else None

    # logger.info(str(model))
    n_parameters = count_parameters(model)
    logger.info(f'Number of params: {n_parameters}')

    if config.MODEL.RESUME:
        load_checkpoint(
            config=config,
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            scaler=scaler,
            logger=logger,
        )

        metric = validate_average(
            config=config,
            model=model,
            fn_metric=fn_metric,
            fn_unpack=fn_unpack,
            loader=loader_val,
            lut=lut_val,
            device=device,
        )

        logger.info(
            f'Loaded weights with metric: {metric}')

    last_epoch = False
    use_time_limit = isinstance(config.TRAIN.TIME_LIMIT, int) and config.TRAIN.TIME_LIMIT > 0
    epoch_start = config.TRAIN.EPOCH_START
    steps_per_epoch = len(loader_train)
    logger.info(
        f'Training for {config.TRAIN.EPOCHS - epoch_start} epoch(s) with {steps_per_epoch} steps per epoch.')

    time_start = time.time()

    for epoch in range(epoch_start, config.TRAIN.EPOCHS):
        train_one_epoch(
            config=config,
            model=model,
            fn_loss=fn_loss,
            fn_metric=fn_metric,
            fn_unpack=fn_unpack,
            loader=loader_train,
            transform=transform,
            mixup=mixup,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            scaler=scaler,
            logger=logger,
            device=device,
            epoch=epoch,
        )

        if epoch == config.TRAIN.EPOCHS - 1:
            last_epoch = True

        if use_time_limit and (time.time() - time_start) >= config.TRAIN.TIME_LIMIT:
            last_epoch = True

        if (epoch + 1) % config.TRAIN.EPOCHS_SAVE == 0 or last_epoch:
            save_checkpoint(
                config=config,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                scaler=scaler,
                logger=logger,
            )

        if (epoch + 1) % config.TRAIN.EPOCHS_EVAL == 0 or last_epoch:
            acc1, acc5 = validate_average(
                config=config,
                model=model,
                fn_metric=fn_metric,
                fn_unpack=fn_unpack,
                loader=loader_val,
                lut=lut_val,
                device=device,
            )

            logger.info(
                f'ACC1-VAL: {acc1:.2f}, ACC5-VAL: {acc5:.2f}')

        so_far = time.time() - time_start
        time_avg = so_far / (epoch - epoch_start + 1)
        eta = time_avg * (config.TRAIN.EPOCHS - epoch - 1)
        so_far = datetime.timedelta(seconds=int(so_far))
        eta = datetime.timedelta(seconds=int(eta))

        logger.info(f'Time -> SO-FAR: {so_far}, ETA: {eta}')

        if last_epoch:
            break

    time_total = time.time() - time_start
    time_total = str(datetime.timedelta(seconds=int(time_total)))
    logger.info('\n' + '#' * 60)
    logger.info(f'Total training time: {time_total}')


def train_one_epoch(
    config, model, fn_loss, fn_metric, fn_unpack, loader, transform,
    mixup, optimizer, lr_scheduler, scaler, logger, device, epoch,
):
    time_start = time.time()
    model.train()
    use_mixup, use_transform = mixup is not None, transform is not None

    meter_metric, meter_loss = Average_Meter(), Average_Meter()
    steps_per_epoch = len(loader)

    for step, batch in enumerate(loader, epoch * steps_per_epoch):
        for param in model.parameters():
            param.grad = None

        inputs, labels = fn_unpack(batch=batch, device=device)

        if use_mixup:
            inputs = mixup(inputs, step=step)

        if use_transform:
            inputs = transform(inputs, step=step)

        if config.TRAIN.AMP:
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = fn_loss(outputs, labels)

            scaler.scale(loss).backward()

            if config.TRAIN.CLIP_GRAD:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(
                    model.parameters(), config.TRAIN.CLIP_GRAD)

            scaler.step(optimizer)
            scaler.update()

        else:
            outputs = model(inputs)
            loss = fn_loss(outputs, labels)

            loss.backward()

            if config.TRAIN.CLIP_GRAD:
                nn.utils.clip_grad_norm_(
                    model.parameters(), config.TRAIN.CLIP_GRAD)

            optimizer.step()

        lr_scheduler.step_update(step)

        metric = fn_metric(outputs.detach().float(), labels)

        batch_size = labels.shape[0]
        meter_metric.update_avg(val=metric.detach(), n=batch_size)
        meter_loss.update_avg(val=loss.item(), n=batch_size)

    time_epoch = time.time() - time_start
    time_epoch = datetime.timedelta(seconds=int(time_epoch))

    logger.info('\n' + '#' * 80)
    logger.info(f'EPOCH [{epoch}/{config.TRAIN.EPOCHS}]')

    acc1, acc5 = meter_metric.avg.cpu().numpy()
    loss, lr = meter_loss.avg, optimizer.param_groups[0]['lr']
    logger.info(
        f'ACC1-TRAIN: {acc1:.2f}, ACC5-TRAIN: {acc5:.2f}, LOSS: {loss:.5f}')
    logger.info(
        f'TIME-TRAIN: {time_epoch}, LR: {lr:.6f}')


@torch.no_grad()
def validate_average(config, model, fn_metric, fn_unpack, loader, lut, device):
    meter_metric = Average_Meter()
    model.eval()

    for batch in loader:
        inputs, idxs_file = fn_unpack(batch=batch, device=device)
        idxs_file = idxs_file.cpu().numpy()

        if config.TRAIN.AMP:
            with torch.cuda.amp.autocast():
                outputs = F.softmax(model(inputs), dim=1)

        else:
            outputs = F.softmax(model(inputs), dim=1)
        # outputs -> [batch_size, n_classes]
        # idxs_file -> [batch_size]

        batch_size = outputs.shape[0]
        idxs_diff = [0]

        for idx in range(1, batch_size):
            if idxs_file[idx] != idxs_file[idx - 1]:
                idxs_diff.append(idx)

        idxs_diff.append(batch_size)
        # idxs_diff -> [n_samples + 1]
        n_samples = len(idxs_diff) - 1

        predictions = [
            outputs[idxs_diff[idx]: idxs_diff[idx + 1]].mean(dim=0) for idx in range(n_samples)]
        predictions = torch.stack(predictions, dim=0)
        # predictions -> [n_samples, n_classes]

        idxs_file = [idxs_file[idx] for idx in idxs_diff[:-1]]
        idxs_file = torch.tensor(idxs_file, device=device).long()
        labels = lut(idxs_file)
        # labels -> [n_samples]

        metric = fn_metric(predictions, labels, act=False)
        meter_metric.update_avg(val=metric, n=n_samples)

    acc1, acc5 = meter_metric.avg.cpu().numpy()

    return acc1, acc5


if __name__ == '__main__':
    main()
