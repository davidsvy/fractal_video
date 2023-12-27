import datetime
import os
import time

import torch
import torch.nn as nn

from cfg._default import get_default_cfg
from src.data import loader_contrastive, unpack_contrastive
from src.ssl import build_ssl_model
from src.transform import transform_contrastive
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
        
    config.TRAIN.MODE = 'contrastive'

    config.freeze()

    main_worker(config)


def main_worker(config):
    if config.MODEL.RESUME:
        assert os.path.isfile(config.MODEL.RESUME)
        
    set_seed(config.SEED)

    config.defrost()

    config.TRAIN.TIME_LIMIT = time_to_secs(config.TRAIN.TIME_LIMIT)
    config.TRAIN.LR_BASE = config.TRAIN.LR_BASE * config.DATA.BATCH_SIZE / 32
    config.TRAIN.LR_WARMUP = config.TRAIN.LR_WARMUP * config.DATA.BATCH_SIZE / 32
    config.TRAIN.LR_MIN = config.TRAIN.LR_MIN * config.DATA.BATCH_SIZE / 32

    config.freeze()

    os.makedirs(config.TRAIN.OUTPUT, exist_ok=True)
    logger = create_logger(config.TRAIN.OUTPUT)
    device = get_device(gpu_id=config.DEVICE, log_fn=logger.info)

    path_config = os.path.join(config.TRAIN.OUTPUT, 'config.yaml')
    with open(path_config, 'w') as file:
        file.write(config.dump())

    logger.info(f'Full config saved to {path_config}')
    logger.info(config.dump())

    dataloader = loader_contrastive(config)
    fn_unpack = unpack_contrastive

    transform = transform_contrastive(config)

    model, fn_loss = build_ssl_model(config)
    model = model.to(device)

    n_params = count_parameters(model)
    logger.info(f'Number of params: {n_params}')

    optimizer = build_optimizer(config=config, model=model)
    lr_scheduler = build_scheduler(
        config=config, optimizer=optimizer, steps_per_epoch=len(dataloader))
    scaler = torch.cuda.amp.GradScaler() if config.TRAIN.AMP else None

    if config.MODEL.RESUME:
        load_checkpoint(
            config=config,
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            scaler=scaler,
            logger=logger,
        )

    last_epoch = False
    use_time_limit = isinstance(
        config.TRAIN.TIME_LIMIT, int) and config.TRAIN.TIME_LIMIT > 0
    epoch_start = config.TRAIN.EPOCH_START
    steps_per_epoch = len(dataloader)
    logger.info(
        f'Training for {config.TRAIN.EPOCHS - epoch_start} epoch(s) with {steps_per_epoch} steps per epoch.')

    time_start = time.time()

    for epoch in range(epoch_start, config.TRAIN.EPOCHS):
        train_one_epoch(
            config=config,
            model=model,
            fn_loss=fn_loss,
            fn_unpack=fn_unpack,
            loader=dataloader,
            transform=transform,
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
    config, model, fn_loss, fn_unpack, loader, transform, optimizer,
    lr_scheduler, scaler, logger, device, epoch,
):
    time_start = time.time()
    model.train()

    use_acc = not (config.TRAIN.SSL_SCHEME.lower() in ['byol'])
    meter_metric, meter_loss = Average_Meter(), Average_Meter()
    steps_per_epoch = len(loader)

    for step, batch in enumerate(loader, epoch * steps_per_epoch):
        for param in model.parameters():
            param.grad = None

        qk, file_idxs = fn_unpack(batch=batch, device=device)
        q, k = transform(qk, step=step)

        if config.TRAIN.AMP:
            with torch.cuda.amp.autocast():
                outputs, labels = model(
                    q=q, k=k, file_idxs=file_idxs, step=step)
                loss = fn_loss(outputs, labels)

            scaler.scale(loss).backward()

            if config.TRAIN.CLIP_GRAD:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(
                    model.parameters(), config.TRAIN.CLIP_GRAD)

            scaler.step(optimizer)
            scaler.update()

        else:
            outputs, labels = model(
                q=q, k=k, file_idxs=file_idxs, step=step)
            loss = fn_loss(outputs, labels)

            loss.backward()

            if config.TRAIN.CLIP_GRAD:
                nn.utils.clip_grad_norm_(
                    model.parameters(), config.TRAIN.CLIP_GRAD)

            optimizer.step()

        lr_scheduler.step_update(step)

        batch_size = labels.shape[0]
        meter_loss.update_avg(val=loss.item(), n=batch_size)

        if use_acc:
            acc = accuracy(
                output=outputs.detach().float(), target=labels, topk=(1, 5))

            meter_metric.update_avg(val=acc.detach(), n=batch_size)

    time_epoch = time.time() - time_start
    time_epoch = datetime.timedelta(seconds=int(time_epoch))

    logger.info('\n' + '#' * 80)
    logger.info(f'EPOCH [{epoch}/{config.TRAIN.EPOCHS}]')
    loss, lr = meter_loss.avg, optimizer.param_groups[0]['lr']

    if use_acc:
        acc1, acc5 = meter_metric.avg.cpu().numpy()
        logger.info(
            f'ACC1-TRAIN: {acc1:.2f}, ACC5-TRAIN: {acc5:.2f}, LOSS: {loss:.5f}')
    else:
        logger.info(f'LOSS: {loss:.5f}')

    logger.info(f'TIME-TRAIN: {time_epoch}, LR: {lr:.6f}')


if __name__ == '__main__':
    main()
