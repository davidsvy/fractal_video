
from src.transform.contrastive import Transform_Contrastive
from src.transform.mixup import Mixup_Background
from src.transform.compose import (
    transform_inner_train,
    transform_inner_val,
    Transform_Outer_Train,
)


def transform_contrastive(config):
    n_steps = max(1, config.AUG.EPOCHS_CURRICULUM * config.STEPS_PER_EPOCH)

    transform = Transform_Contrastive(
        img_size=config.DATA.IMG_SIZE,
        easy_k=config.AUG.SSL_EASY_K,
        randaugment_m=config.AUG.AUTO_AUGMENT_M,
        randaugment_n=config.AUG.AUTO_AUGMENT_N,
        n_steps=n_steps,
        back=config.AUG.TYPE_MIXUP == 'back',
        prob_perspective=config.AUG.PROB_PERSPECTIVE,
        prob_scale=config.AUG.PROB_SCALE,
        prob_shift=config.AUG.PROB_SHIFT,
        prob_clone=config.AUG.PROB_CLONE,
        prob_zoom=config.AUG.PROB_ZOOM,
        prob_shake=config.AUG.PROB_SHAKE,
    )

    return transform


##########################################################################
##########################################################################
# OTHER
##########################################################################
##########################################################################

def transform_inner(is_train, config):
    if is_train:
        transform = transform_inner_train(
            crop_size=config.DATA.IMG_SIZE,
            min_scale=config.AUG.MIN_SCALE,
            interp=config.AUG.INTERP,
        )

    else:
        transform = transform_inner_val(
            crop_size=config.DATA.IMG_SIZE,
            interp=config.AUG.INTERP,
            crop_inc=config.AUG.CROP_VAL,
        )

    return transform


def transform_outer_train(config):
    n_steps = max(1, config.AUG.EPOCHS_CURRICULUM * config.STEPS_PER_EPOCH)

    transform = Transform_Outer_Train(
        randaugment_m=config.AUG.AUTO_AUGMENT_M,
        randaugment_n=config.AUG.AUTO_AUGMENT_N,
        n_steps=n_steps,
        hor_flip=config.AUG.HOR_FLIP,
        prob_blur=config.AUG.PROB_BLUR,
        prob_perspective=config.AUG.PROB_PERSPECTIVE,
        prob_shift=config.AUG.PROB_SHIFT,
        prob_zoom=config.AUG.PROB_ZOOM,
        prob_shake=config.AUG.PROB_SHAKE,
    )

    return transform


def build_mixup(config):
    mixup = None

    if config.AUG.TYPE_MIXUP == 'back':
        n_steps = max(1, config.AUG.EPOCHS_CURRICULUM * config.STEPS_PER_EPOCH)

        mixup = Mixup_Background(
            img_size=config.DATA.IMG_SIZE,
            n_steps=n_steps,
            prob_scale=config.AUG.PROB_SCALE,
            prob_shift=config.AUG.PROB_SHIFT,
            prob_clone=config.AUG.PROB_CLONE,
        )

    return mixup
