from yacs.config import CfgNode as CN

_C = CN()

# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
_C.DATA = CN()
_C.DATA.IMG_SIZE = 112
_C.DATA.CLIP_LENGTH = 8
_C.DATA.N_CHANNELS = 3
_C.DATA.ROOT = 'data'
# Dataset must be stores at ROOT/DATASET
_C.DATA.DATASET = 'hmdb51'
_C.DATA.EXTENSION = 'avi'
_C.DATA.BATCH_SIZE = 32
_C.DATA.BATCH_SIZE_VAL = 4
_C.DATA.STRIDE = 6
_C.DATA.VAL_MIN_STEPS = 1
_C.DATA.VAL_MAX_SEGS = 10
_C.DATA.DECORD_N_THREADS = 1
_C.DATA.TORCH_N_WORKERS = 8

# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.OUTPUT = 'out'
# Self-supervised training objective. Select from ['moco', 'byol', 'simclr']
_C.TRAIN.SSL_SCHEME = 'moco'
_C.TRAIN.EPOCH_START = 0
_C.TRAIN.EPOCHS = 100
_C.TRAIN.EPOCHS_WARMUP = 10
_C.TRAIN.EPOCHS_SAVE = 50
_C.TRAIN.EPOCHS_EVAL = 10

_C.TRAIN.WEIGHT_DECAY = 0.01
_C.TRAIN.LR_BASE = 0.0008
_C.TRAIN.LR_WARMUP = 0.000001
_C.TRAIN.LR_MIN = 0.00001
_C.TRAIN.CLIP_GRAD = 5.0
# LR scheduler
_C.TRAIN.LR_SCHEDULER = CN()
_C.TRAIN.LR_SCHEDULER.NAME = 'cosine'
# Step interval to decay LR, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.EPOCHS_DECAY = 12
# LR decay rate, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_RATE = 0.1
# Optimizer
_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.NAME = 'adamw'
# Optimizer Epsilon
_C.TRAIN.OPTIMIZER.EPS = 1e-8
# Optimizer Betas
_C.TRAIN.OPTIMIZER.BETAS = (0.9, 0.999)
# SGD momentum
_C.TRAIN.OPTIMIZER.MOMENTUM = 0.9
# Path to self-supervised checkpoint to initialize model weights.
_C.TRAIN.INIT_SSL = None
# Path to supervised checkpoint to initialize model weights.
_C.TRAIN.INIT_SUP = None

# Mixed Precision
_C.TRAIN.AMP = True
# Finish training after TIME_LIMIT, eg 6h40m.
_C.TRAIN.TIME_LIMIT = None
# Maximum number of checkpoints to store during training. Only N_CHECK most recent checkpoints are kept.
_C.TRAIN.N_CHECK = 5


# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# Path to checkpoint to continue training & initialize model & optimizer weights from checkpoint.
_C.MODEL.RESUME = None
_C.MODEL.ARCH = 'tsm'
# Path to Kinetics checkpoint.
_C.MODEL.KINETICS = None

_C.MODEL.X3D_SIZE = 'L'  # ['M', 'L', 'XL']
_C.MODEL.TSM_BASE = 'resnet50'  # ['resnet50', 'mobilenetv2']


# -----------------------------------------------------------------------------
# MoCo settings
# -----------------------------------------------------------------------------
_C.MOCO = CN()
_C.MOCO.DIM = 128
_C.MOCO.K = 2 ** 14
_C.MOCO.M = 0.999
_C.MOCO.T = 0.07
_C.MOCO.MLP_HEAD = True


# -----------------------------------------------------------------------------
# SimCLR settings
# -----------------------------------------------------------------------------
_C.SIMCLR = CN()
_C.SIMCLR.DIM = 128
_C.SIMCLR.T = 0.1
_C.SIMCLR.MLP_HEAD = True


# -----------------------------------------------------------------------------
# BYOL settings
# -----------------------------------------------------------------------------
_C.BYOL = CN()
_C.BYOL.D_OUTPUT = 256
_C.BYOL.D_HIDDEN = 1024
_C.BYOL.EMA_WEIGHT = 0.997


# -----------------------------------------------------------------------------
# Augmentation settings
# -----------------------------------------------------------------------------
_C.AUG = CN()
_C.AUG.INTERP = 'bicubic'
_C.AUG.HOR_FLIP = True
_C.AUG.CROP_VAL = True
_C.AUG.MIN_SCALE = 0.2

_C.AUG.AUTO_AUGMENT_M = 7
_C.AUG.AUTO_AUGMENT_N = 2

_C.AUG.TYPE_MIXUP = 'none'  # ['none', 'back']

_C.AUG.EPOCHS_CURRICULUM = 5
_C.AUG.SSL_EASY_K = True

# Probabilities for different augmentations.
_C.AUG.PROB_BLUR = 0.5
_C.AUG.PROB_SCALE = 0.0
_C.AUG.PROB_SHIFT = 0.0
_C.AUG.PROB_SHAKE = 0.0
_C.AUG.PROB_ZOOM = 0.0
_C.AUG.PROB_PERSPECTIVE = 0.0
_C.AUG.PROB_CLONE = 0.0


# -----------------------------------------------------------------------------
# Misc
# -----------------------------------------------------------------------------
_C.SEED = 1
_C.DEVICE = 0


def get_default_cfg():
    """Get a yacs CfgNode object with default values."""
    config = _C.clone()
    config.freeze()

    return config
