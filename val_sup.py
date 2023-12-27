import datetime
import numpy as np
import os
import pandas
import time

import torch
import torch.nn.functional as F

from cfg._default import get_default_cfg
from src.data import loader_test, unpack_labeled
from src.encoder import build_encoder
from src.utils.args import parse_args_train
from src.utils.other import Average_Meter, set_seed

from src.utils.torch import (
    accuracy, 
    count_parameters, 
    get_device, 
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
    assert os.path.isfile(config.MODEL.RESUME)
    set_seed(config.SEED)

    os.makedirs(config.TRAIN.OUTPUT, exist_ok=True)
    device = get_device(gpu_id=config.DEVICE)

    path_config = os.path.join(config.TRAIN.OUTPUT, 'config_test.yaml')
    with open(path_config, 'w') as file:
        file.write(config.dump())

    dataloader, lut_label = loader_test(config)
    fn_unpack = unpack_labeled
    fn_metric = accuracy

    if lut_label is not None:
        lut_label = lut_label.to(device)

    ckpt = torch.load(config.MODEL.RESUME, map_location='cpu')
    model = build_encoder(config=ckpt['config'], mlp_head=False)
    model = model.to(device)

    n_parameters = count_parameters(model)
    print(f'Number of params: {n_parameters}')

    msg = model.load_state_dict(ckpt['model'], strict=False)
    print(msg)

    time_start = time.time()

    acc1, acc5 = test(
        config=config,
        model=model,
        fn_metric=fn_metric,
        fn_unpack=fn_unpack,
        loader=dataloader,
        lut=lut_label,
        device=device,
    )

    print(
        f'ACC1-VAL: {acc1:.2f}, ACC5-VAL: {acc5:.2f}')

    time_total = time.time() - time_start
    time_total = str(datetime.timedelta(seconds=int(time_total)))
    print('\n' + '#' * 60)
    print(f'Total duration: {time_total}')


@torch.no_grad()
def test(config, model, fn_metric, fn_unpack, loader, lut, device):
    meter_metric = Average_Meter()
    model.eval()

    dict_csv = {
        'path': [],
        'label': [],
        'pred1': [],
        'pred2': [],
        'pred3': [],
    }

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

        idxs_file, labels = idxs_file.cpu().numpy(), labels.cpu().numpy()
        # idxs_file, labels -> [n_samples]
        idxs_pred = torch.topk(
            predictions, k=3, dim=1, largest=True, sorted=True)
        idxs_pred = idxs_pred.indices.cpu().numpy()
        # idxs_pred -> [n_samples, 3]

        dict_csv['path'].append(idxs_file)
        dict_csv['label'].append(labels)
        dict_csv['pred1'].append(idxs_pred[:, 0])
        dict_csv['pred2'].append(idxs_pred[:, 1])
        dict_csv['pred3'].append(idxs_pred[:, 2])

    for key in dict_csv:
        dict_csv[key] = np.concatenate(dict_csv[key], axis=0)

    path_csv = os.path.join(config.TRAIN.OUTPUT, 'pred_test.csv')
    df_csv = pandas.DataFrame(dict_csv)
    df_csv.to_csv(path_csv, sep=' ', header=True, index=False)

    acc1, acc5 = meter_metric.avg.cpu().numpy()

    return acc1, acc5


if __name__ == '__main__':
    main()
