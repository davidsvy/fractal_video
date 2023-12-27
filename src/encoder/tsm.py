# Adapted from:
# https://github.com/mit-han-lab/temporal-shift-module#pretrained-models
import math
import os

import torch
import torch.nn as nn
import torchvision


###################################################################################
# MOBILENET
###################################################################################


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def make_divisible(x, divisible_by=8):
    import numpy as np
    return int(np.ceil(x * 1. / divisible_by) * divisible_by)


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride,
                          1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride,
                          1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, n_class=1000, input_size=224, width_mult=1.):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        assert input_size % 32 == 0
        # input_channel = make_divisible(input_channel * width_mult)  # first channel is always 32!
        self.last_channel = make_divisible(
            last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(3, input_channel, 2)]
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = make_divisible(c * width_mult) if t > 1 else c
            for i in range(n):
                if i == 0:
                    self.features.append(
                        block(input_channel, output_channel, s, expand_ratio=t))
                else:
                    self.features.append(
                        block(input_channel, output_channel, 1, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        self.features.append(conv_1x1_bn(input_channel, self.last_channel))
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building classifier
        self.classifier = nn.Linear(self.last_channel, n_class)

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.mean(3).mean(2)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def mobilenet_v2(pretrained=True):
    model = MobileNetV2(width_mult=1)

    if pretrained:
        try:
            from torch.hub import load_state_dict_from_url
        except ImportError:
            from torch.utils.model_zoo import load_url as load_state_dict_from_url
        state_dict = load_state_dict_from_url(
            'https://www.dropbox.com/s/47tyzpofuuyyv1b/mobilenetv2_1.0-f2a8633.pth.tar?dl=1', progress=True)
        model.load_state_dict(state_dict)
    return model

###################################################################################
# CONSENSUS
###################################################################################


class Identity(torch.nn.Module):
    def forward(self, input):
        return input


class SegmentConsensus(torch.nn.Module):

    def __init__(self, consensus_type, dim=1):
        super(SegmentConsensus, self).__init__()
        self.consensus_type = consensus_type
        self.dim = dim
        self.shape = None

    def forward(self, input_tensor):
        self.shape = input_tensor.size()
        if self.consensus_type == 'avg':
            output = input_tensor.mean(dim=self.dim, keepdim=True)
        elif self.consensus_type == 'identity':
            output = input_tensor
        else:
            output = None

        return output


class ConsensusModule(torch.nn.Module):

    def __init__(self, consensus_type, dim=1):
        super(ConsensusModule, self).__init__()
        self.consensus_type = consensus_type if consensus_type != 'rnn' else 'identity'
        self.dim = dim

    def forward(self, input):
        return SegmentConsensus(self.consensus_type, self.dim)(input)

###################################################################################
# SHIFT
###################################################################################


class TemporalShift(nn.Module):
    def __init__(self, net, n_segment=3, n_div=8, inplace=False):
        super(TemporalShift, self).__init__()
        self.net = net
        self.n_segment = n_segment
        self.fold_div = n_div
        self.inplace = inplace
        if inplace:
            print('=> Using in-place shift...')

        print('=> Using fold div: {}'.format(self.fold_div))

    def forward(self, x):
        x = self.shift(
            x, self.n_segment, fold_div=self.fold_div, inplace=self.inplace)

        return self.net(x)

    @staticmethod
    def shift(x, n_segment, fold_div=3, inplace=False):
        nt, c, h, w = x.size()
        n_batch = nt // n_segment
        x = x.view(n_batch, n_segment, c, h, w)

        fold = c // fold_div
        if inplace:
            # Due to some out of order error when performing parallel computing.
            # May need to write a CUDA kernel.
            raise NotImplementedError
            # out = InplaceShift.apply(x, fold)
        else:
            out = torch.zeros_like(x)
            out[:, :-1, :fold] = x[:, 1:, :fold]  # shift left
            out[:, 1:, fold: 2 * fold] = x[
                :, :-1, fold: 2 * fold]  # shift right
            out[:, :, 2 * fold:] = x[:, :, 2 * fold:]  # not shift

        return out.view(nt, c, h, w)


def make_temporal_shift(net, n_segment, n_div=8, place='blockres', temporal_pool=False):
    if temporal_pool:
        n_segment_list = [
            n_segment, n_segment // 2, n_segment // 2, n_segment // 2]
    else:
        n_segment_list = [n_segment] * 4
    assert n_segment_list[-1] > 0
    print('=> n_segment per stage: {}'.format(n_segment_list))

    if isinstance(net, torchvision.models.ResNet):
        if place == 'block':
            def make_block_temporal(stage, this_segment):
                blocks = list(stage.children())
                print('=> Processing stage with {} blocks'.format(len(blocks)))
                for i, b in enumerate(blocks):
                    blocks[i] = TemporalShift(
                        b, n_segment=this_segment, n_div=n_div)

                return nn.Sequential(*(blocks))

            net.layer1 = make_block_temporal(net.layer1, n_segment_list[0])
            net.layer2 = make_block_temporal(net.layer2, n_segment_list[1])
            net.layer3 = make_block_temporal(net.layer3, n_segment_list[2])
            net.layer4 = make_block_temporal(net.layer4, n_segment_list[3])

        elif 'blockres' in place:
            n_round = 1
            if len(list(net.layer3.children())) >= 23:
                n_round = 2
                print('=> Using n_round {} to insert temporal shift'.format(n_round))

            def make_block_temporal(stage, this_segment):
                blocks = list(stage.children())
                print('=> Processing stage with {} blocks residual'.format(len(blocks)))
                for i, b in enumerate(blocks):
                    if i % n_round == 0:
                        blocks[i].conv1 = TemporalShift(
                            b.conv1, n_segment=this_segment, n_div=n_div)

                return nn.Sequential(*blocks)

            net.layer1 = make_block_temporal(net.layer1, n_segment_list[0])
            net.layer2 = make_block_temporal(net.layer2, n_segment_list[1])
            net.layer3 = make_block_temporal(net.layer3, n_segment_list[2])
            net.layer4 = make_block_temporal(net.layer4, n_segment_list[3])
    else:
        raise NotImplementedError(place)

###################################################################################
# TSM
###################################################################################


class TSM(nn.Module):
    def __init__(
        self,
        num_class,
        num_segments,
        base_model='resnet50',
        consensus_type='avg',
        before_softmax=True,
        dropout=0.8,
        partial_bn=False,
        print_spec=True,
        imagenet=False,
        is_shift=True,
        shift_div=8,
        shift_place='blockres',
        temporal_pool=False,
    ):
        super(TSM, self).__init__()
        self.new_length = 1
        self.num_segments = num_segments
        self.reshape = True
        self.before_softmax = before_softmax
        self.dropout = dropout
        self.consensus_type = consensus_type
        # the dimension of the CNN feature to represent each frame
        self.imagenet = imagenet

        self.is_shift = is_shift
        self.shift_div = shift_div
        self.shift_place = shift_place
        self.base_model_name = base_model
        self.temporal_pool = temporal_pool

        if not before_softmax and consensus_type != 'avg':
            raise ValueError("Only avg consensus can be used after Softmax")

        if print_spec:
            print((f"""
    Initializing TSN with base model: {base_model}.
    TSN Configurations:
        num_segments:       {self.num_segments}
        new_length:         {self.new_length}
        consensus_module:   {consensus_type}
        dropout_ratio:      {self.dropout}
            """))

        self._prepare_base_model(base_model)

        feature_dim = self._prepare_tsn(num_class)

        self.consensus = ConsensusModule(consensus_type)

        if not self.before_softmax:
            self.softmax = nn.Softmax()

        self._enable_pbn = partial_bn
        if partial_bn:
            self.partialBN(True)

    def _prepare_tsn(self, num_class):
        feature_dim = getattr(
            self.base_model, self.base_model.last_layer_name).in_features

        if self.dropout == 0:
            setattr(
                self.base_model, self.base_model.last_layer_name, nn.Linear(feature_dim, num_class))
            self.new_fc = None
        else:
            setattr(
                self.base_model, self.base_model.last_layer_name, nn.Dropout(p=self.dropout))
            self.new_fc = nn.Linear(feature_dim, num_class)

        std = 0.001
        if self.new_fc is None:
            nn.init.normal_(
                getattr(self.base_model, self.base_model.last_layer_name).weight, 0, std)
            nn.init.constant_(
                getattr(self.base_model, self.base_model.last_layer_name).bias, 0)

        elif hasattr(self.new_fc, 'weight'):
            nn.init.normal_(self.new_fc.weight, 0, std)
            nn.init.constant_(self.new_fc.bias, 0)

        return feature_dim

    def _prepare_base_model(self, base_model):
        self.input_size = 224
        self.input_mean = [0.485, 0.456, 0.406]
        self.input_std = [0.229, 0.224, 0.225]

        print('=> base model: {}'.format(base_model))

        if base_model == 'resnet50':
            self.base_model = torchvision.models.resnet50(
                pretrained=self.imagenet)

            if self.is_shift:
                print('Adding temporal shift...')
                make_temporal_shift(
                    net=self.base_model,
                    n_segment=self.num_segments,
                    n_div=self.shift_div,
                    place=self.shift_place,
                    temporal_pool=self.temporal_pool,
                )

            self.base_model.last_layer_name = 'fc'
            self.base_model.avgpool = nn.AdaptiveAvgPool2d(1)

        elif base_model == 'mobilenetv2':
            self.base_model = mobilenet_v2(self.imagenet)

            self.base_model.last_layer_name = 'classifier'
            self.base_model.avgpool = nn.AdaptiveAvgPool2d(1)

            if self.is_shift:
                for m in self.base_model.modules():
                    if isinstance(m, InvertedResidual) and len(m.conv) == 8 and m.use_res_connect:
                        print(
                            f'Adding temporal shift... {m.use_res_connect}')

                        m.conv[0] = TemporalShift(
                            net=m.conv[0],
                            n_segment=self.num_segments,
                            n_div=self.shift_div,
                        )

        else:
            raise ValueError(f'Unknown base model: {base_model}')

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        :return:
        """
        super(TSM, self).train(mode)
        count = 0

        if self._enable_pbn and mode:
            print("Freezing BatchNorm2D except the first one.")
            for m in self.base_model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    count += 1
                    if count >= (2 if self._enable_pbn else 1):
                        m.eval()
                        # shutdown update in frozen mode
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False

    def partialBN(self, enable):
        self._enable_pbn = enable

    def forward(self, input):
        # input -> [B, C, T, H, W]
        B, C, T, H, W = input.shape
        input = input.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        # input -> [B * T, C, H, W]

        base_out = self.base_model(input)

        if self.dropout > 0:
            base_out = self.new_fc(base_out)

        if not self.before_softmax:
            base_out = self.softmax(base_out)

        if self.is_shift and self.temporal_pool:
            base_out = base_out.view(
                (-1, self.num_segments // 2) + base_out.size()[1:])
        else:
            base_out = base_out.view(
                (-1, self.num_segments) + base_out.size()[1:])
        output = self.consensus(base_out)
        return output.squeeze(1)

    @property
    def crop_size(self):
        return self.input_size

    @property
    def scale_size(self):
        return self.input_size * 256 // 224


def kinetics(config, model):
    """Manually download weights:

    resnet50:
        https://hanlab18.mit.edu/projects/tsm/models/TSM_kinetics_RGB_resnet50_shift8_blockres_avg_segment8_e100_dense.pth

    mobilenetv2:
        https://hanlab18.mit.edu/projects/tsm/models/TSM_kinetics_RGB_mobilenetv2_shift8_blockres_avg_segment8_e100_dense.pth

    """
    path_ckpt = config.MODEL.KINETICS
    if not os.path.isfile(path_ckpt):
        raise ValueError(
            f'No KINETICS checkpoint found at {path_ckpt}.')

    ckpt = torch.load(path_ckpt, map_location='cpu')['state_dict']

    if config.MODEL.TSM_BASE == 'resnet50':
        names_param = list(ckpt.keys())
        for name in names_param:
            ckpt[name[7:]] = ckpt.pop(name)

    print(f'Using KINETICS checkpoint at {path_ckpt}.')
    msg = model.load_state_dict(ckpt, strict=False)
    print(msg)


def build(config, mlp_head=False, *args, **kwargs):
    model = TSM(
        num_class=400,
        num_segments=config.DATA.CLIP_LENGTH,
        base_model=config.MODEL.TSM_BASE,  # ['resnet50', 'mobilenetv2']
        imagenet=False,
    )

    if config.MODEL.KINETICS is not None:
        kinetics(config=config, model=model)

    in_features = model.new_fc.in_features
    if mlp_head:
        d_hidden = 512
        model.new_fc = nn.Sequential(
            nn.Linear(in_features, d_hidden),
            nn.SiLU(),
            nn.Linear(d_hidden, config.MODEL.N_CLASSES),
        )

    else:
        model.new_fc = nn.Linear(in_features, config.MODEL.N_CLASSES)

    return model


def init_head(model, n_classes):
    old_head = model.new_fc

    if isinstance(old_head, nn.Linear):
        in_features = old_head.in_features

    elif isinstance(old_head, nn.Sequential):
        in_features = old_head[0].in_features

    else:
        raise ValueError(f'Class of head is invalid: {type(old_head)}')

    model.new_fc = nn.Linear(in_features, n_classes)
    del old_head

    return model
