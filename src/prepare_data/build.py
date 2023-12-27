import datetime
import time

from .diving48 import diving48_preprocessed
from .egtea import egtea_preprocessed
from .gdrive import gdrive_supervised, gdrive_unsupervised
from .hmdb51 import hmdb51
from .scenes import scenes
from .ucf101 import ucf101
from .volleyball import volleyball_preprocessed


def prepare_data(dataset, root='data'):
    time_start = time.time()

    if dataset == 'diving48':
        diving48_preprocessed(root)

    elif dataset == 'scenes':
        scenes(root)

    elif dataset == 'egtea':
        egtea_preprocessed(root)

    elif dataset == 'hmdb51':
        hmdb51(root)

    elif dataset.startswith(('sup',)):
        gdrive_supervised(dataset=dataset, root=root)

    elif dataset == 'ucf101':
        ucf101(root)

    elif dataset.startswith(('uns',)):
        gdrive_unsupervised(dataset=dataset, root=root)

    elif dataset == 'volleyball':
        volleyball_preprocessed(root)

    else:
        raise ValueError(f'Unknown dataset: {dataset}')

    time_run = datetime.timedelta(seconds=int(time.time() - time_start))
    print('#' * 50)
    print(f'Total time: {time_run}')
