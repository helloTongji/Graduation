"""
Code source: https://github.com/pytorch/vision
"""
from __future__ import division, absolute_import
import torch.utils.model_zoo as model_zoo
from torch import nn
import math
from torch.autograd import Function
import torchvision.datasets as dsets
from torchvision import transforms
from torch.autograd import Variable
import warnings
import torch
from torch import nn
from torch.nn import functional as F
# from .squeezenet import *
import math

import copy
from torchvision.models.resnet import resnet50, Bottleneck


__all__ = [
    'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
    'resnext50_32x4d', 'resnext101_32x8d', 'resnet50_fc512'
]

model_urls = {
    'resnet18':
    'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34':
    'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50':
    'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101':
    'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152':
    'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d':
    'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d':
    'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=1, stride=stride, bias=False
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None
    ):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError(
                'BasicBlock only supports groups=1 and base_width=64'
            )
        if dilation > 1:
            raise NotImplementedError(
                "Dilation > 1 not supported in BasicBlock"
            )
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None
    ):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width/64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """Residual network.
    
    Reference:
        - He et al. Deep Residual Learning for Image Recognition. CVPR 2016.
        - Xie et al. Aggregated Residual Transformations for Deep Neural Networks. CVPR 2017.

    Public keys:
        - ``resnet18``: ResNet18.
        - ``resnet34``: ResNet34.
        - ``resnet50``: ResNet50.
        - ``resnet101``: ResNet101.
        - ``resnet152``: ResNet152.
        - ``resnext50_32x4d``: ResNeXt50.
        - ``resnext101_32x8d``: ResNeXt101.
        - ``resnet50_fc512``: ResNet50 + FC.
    """

    def __init__(
        self,
        num_classes,
        loss,
        block,
        layers,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=None,
        last_stride=2,
        fc_dims=None,
        dropout_p=None,
        **kwargs
    ):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
            
            
        self.hashbit=256
        self._norm_layer = norm_layer
        self.loss = loss
        self.feature_dim = 512 * block.expansion
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".
                format(replace_stride_with_dilation)
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(
            3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(
            block,
            128,
            layers[1],
            stride=2,
            dilate=replace_stride_with_dilation[0]
        )
        self.layer3 = self._make_layer(
            block,
            256,
            layers[2],
            stride=2,
            dilate=replace_stride_with_dilation[1]
        )
        self.layer4 = self._make_layer(
            block,
            512,
            layers[3],
            stride=last_stride,
            dilate=replace_stride_with_dilation[2]
        )
        self.global_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = self._construct_fc_layer(
            fc_dims, 512 * block.expansion, dropout_p
        )
        self.classifier_resnet = nn.Linear(self.feature_dim, num_classes)
        # self.classifier = nn.Linear(self.feature_dim, num_classes)

        #self._init_params()

        
        
        
        self.classifier = nn.Linear(self.hashbit, num_classes)
        # self.fc_encode = nn.Linear(self.feature_dim, 128)
        
        #self.
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        feats=256
        self.backbone = nn.Sequential(
            self.conv1,
            self.bn1,
            self.relu,
            self.maxpool,
            self.layer1,
            self.layer2,
            self.layer3[0],
        )
        res_conv4 = nn.Sequential(*self.layer3[1:])

        # res_g_conv5 = resnet.layer4

        res_p_conv5 = nn.Sequential(
            Bottleneck(1024, 512, downsample=nn.Sequential(nn.Conv2d(1024, 2048, 1, bias=False), nn.BatchNorm2d(2048))),
            Bottleneck(2048, 512),
            Bottleneck(2048, 512))
        res_p_conv5.load_state_dict(self.layer4.state_dict())

        # self.p1 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_g_conv5))
        self.p3 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_p_conv5))
        self.maxpool_zg_p1 = nn.MaxPool2d(kernel_size=(12, 4))
        self.maxpool_zg_p3 = nn.MaxPool2d(kernel_size=(24, 8))
        self.maxpool_zp3 = nn.MaxPool2d(kernel_size=(8, 8))

        self.reduction = nn.Sequential(nn.Conv2d(2048, feats, 1, bias=False), nn.BatchNorm2d(feats), nn.ReLU())

        self._init_reduction(self.reduction)

        # self.fc_id_2048_0 = nn.Linear(feats, num_classes)
        self.fc_id_2048_2 = nn.Linear(feats, num_classes)

        self.fc_id_256_2_0 = nn.Linear(feats, num_classes)
        self.fc_id_256_2_1 = nn.Linear(feats, num_classes)
        self.fc_id_256_2_2 = nn.Linear(feats, num_classes)

        # self._init_fc(self.fc_id_2048_0)
        self._init_fc(self.fc_id_2048_2)

        self._init_fc(self.fc_id_256_2_0)
        self._init_fc(self.fc_id_256_2_1)
        self._init_fc(self.fc_id_256_2_2)
        self._init_params()


        self.split_num=10
        self.fc_split = nn.Linear(feats*3+self.feature_dim, self.hashbit*self.split_num)
        
        self.de1 = DivideEncode(self.hashbit*self.split_num, self.split_num)
        

        
        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups,
                self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer
                )
            )

        return nn.Sequential(*layers)

    def _construct_fc_layer(self, fc_dims, input_dim, dropout_p=None):
        """Constructs fully connected layer

        Args:
            fc_dims (list or tuple): dimensions of fc layers, if None, no fc layers are constructed
            input_dim (int): input dimension
            dropout_p (float): dropout probability, if None, dropout is unused
        """
        if fc_dims is None:
            self.feature_dim = input_dim
            return None

        assert isinstance(
            fc_dims, (list, tuple)
        ), 'fc_dims must be either list or tuple, but got {}'.format(
            type(fc_dims)
        )

        layers = []
        for dim in fc_dims:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout_p is not None:
                layers.append(nn.Dropout(p=dropout_p))
            input_dim = dim

        self.feature_dim = fc_dims[-1]

        return nn.Sequential(*layers)

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu'
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


    def featuremaps(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x
        
    #MGN method
    @staticmethod
    def _init_reduction(reduction):
        # conv
        nn.init.kaiming_normal_(reduction[0].weight, mode='fan_in')
        # nn.init.constant_(reduction[0].bias, 0.)

        # bn
        nn.init.normal_(reduction[1].weight, mean=1., std=0.02)
        nn.init.constant_(reduction[1].bias, 0.)

    @staticmethod
    def _init_fc(fc):
        nn.init.kaiming_normal_(fc.weight, mode='fan_out')
        # nn.init.normal_(fc.weight, std=0.001)
        nn.init.constant_(fc.bias, 0.)
        
    #MGN method
    
    
    
    def forward(self, x):
        
        f = self.featuremaps(x)
        v_resnet = self.global_avgpool(f)
        v_resnet = v_resnet.view(v_resnet.size(0), -1)
        if self.fc is not None:
            v_resnet = self.fc(v_resnet)
        y_resnet = self.classifier_resnet(v_resnet)



        
        
        
        x_mgn = self.backbone(x)
        p3 = self.p3(x_mgn)

        zp3 = self.maxpool_zp3(p3)
        z0_p3 = zp3[:, :, 0:1, :]
        z1_p3 = zp3[:, :, 1:2, :]
        z2_p3 = zp3[:, :, 2:3, :]
        f0_p3 = self.reduction(z0_p3).squeeze(dim=3).squeeze(dim=2)
        f1_p3 = self.reduction(z1_p3).squeeze(dim=3).squeeze(dim=2)
        f2_p3 = self.reduction(z2_p3).squeeze(dim=3).squeeze(dim=2)
        # print("f0_p3:",f0_p3.shape)
        l0_p3 = self.fc_id_256_2_0(f0_p3)
        l1_p3 = self.fc_id_256_2_1(f1_p3)
        l2_p3 = self.fc_id_256_2_2(f2_p3)
        #mgn_part=torch.cat([f0_p3,f1_p3,f2_p3,l0_p3,l1_p3,l2_p3],dim=1)
        #feats_mgn=torch.cat([f0_p3,f1_p3,f2_p3],dim=1)

        v=torch.cat([v_resnet,f0_p3,f1_p3,f2_p3],dim=1)
        
        
        v_split=self.fc_split(v)
        h_return=self.de1(v_split)
        y = self.classifier(h_return)
        b_return=hash_layer(h_return)
        
        if self.loss == 'softmax':
            return y
        elif self.loss == 'triplet':
            return y, v,h_return,b_return,y_resnet,l0_p3,l1_p3,l2_p3#,fg_p3#,b_weighted#,y_pcb,v_pcb_return
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))










class DivideEncode(nn.Module):
    '''
    Implementation of the divide-and-encode module in,
    Simultaneous Feature Learning and Hash Coding with Deep Neural Networks
    https://arxiv.org/pdf/1504.03410.pdf
    '''
    def __init__(self, num_inputs, num_per_group):
        super().__init__()
        assert num_inputs % num_per_group == 0, \
            "num_per_group should be divisible by num_inputs."
        self.num_groups = num_inputs // num_per_group
        self.num_per_group = num_per_group
        weights_dim = (self.num_groups, self.num_per_group)
        self.weights = nn.Parameter(torch.empty(weights_dim))
        nn.init.xavier_normal_(self.weights)

    def forward(self, X):
        X = X.view((-1, self.num_groups, self.num_per_group))
        return X.mul(self.weights).sum(2)

import numpy as np
# new layer
class hash(Function):
    @staticmethod
    def forward(ctx, input):
        return torch.sign(input)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def hash_layer(input):
    return hash.apply(input)  














class ConvBlock(nn.Module):
    """Basic convolutional block.
    
    convolution + batch normalization + relu.

    Args:
        in_c (int): number of input channels.
        out_c (int): number of output channels.
        k (int or tuple): kernel size.
        s (int or tuple): stride.
        p (int or tuple): padding.
    """

    def __init__(self, in_c, out_c, k, s=1, p=0):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, stride=s, padding=p)
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))

class SpatialAttn(nn.Module):
    """Spatial Attention (Sec. 3.1.I.1)"""

    def __init__(self):
        super(SpatialAttn, self).__init__()
        self.conv1 = ConvBlock(1, 1, 3, s=2, p=1)
        self.conv2 = ConvBlock(1, 1, 1)

    def forward(self, x):
        # global cross-channel averaging
        # print("x",x.shape)
        x = x.mean(1, keepdim=True)
        # 3-by-3 conv
        x = self.conv1(x)
        # print(x.shape)
        # bilinear resizing
        x = F.upsample(
            x, (x.size(2) +1, x.size(3)),
            mode='bilinear',
            align_corners=True
        )
        # print("up:",x.shape)
        # scaling conv
        x = self.conv2(x)
        return x


class ChannelAttn(nn.Module):
    """Channel Attention (Sec. 3.1.I.2)"""

    def __init__(self, in_channels, reduction_rate=16):
        super(ChannelAttn, self).__init__()
        assert in_channels % reduction_rate == 0
        self.conv1 = ConvBlock(in_channels, in_channels // reduction_rate, 1)
        self.conv2 = ConvBlock(in_channels // reduction_rate, in_channels, 1)

    def forward(self, x):
        # squeeze operation (global average pooling)
        x = F.avg_pool2d(x, x.size()[2:])
        # excitation operation (2 conv layers)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class SoftAttn(nn.Module):
    """Soft Attention (Sec. 3.1.I)
    
    Aim: Spatial Attention + Channel Attention
    
    Output: attention maps with shape identical to input.
    """

    def __init__(self, in_channels):
        super(SoftAttn, self).__init__()
        self.spatial_attn = SpatialAttn()
        self.channel_attn = ChannelAttn(in_channels)
        self.conv = ConvBlock(in_channels, in_channels, 1)

    def forward(self, x):
        y_spatial = self.spatial_attn(x)

        y_channel = self.channel_attn(x)
        
        print(y_spatial.shape)
        print(y_channel.shape)
        y = y_spatial * y_channel
        y = torch.sigmoid(self.conv(y))
        return y


class HardAttn(nn.Module):
    """Hard Attention (Sec. 3.1.II)"""

    def __init__(self, in_channels):
        super(HardAttn, self).__init__()
        self.fc = nn.Linear(in_channels, 4 * 2)
        self.init_params()

    def init_params(self):
        self.fc.weight.data.zero_()
        self.fc.bias.data.copy_(
            torch.tensor(
                [0, -0.75, 0, -0.25, 0, 0.25, 0, 0.75], dtype=torch.float
            )
        )

    def forward(self, x):
        # squeeze operation (global average pooling)
        x = F.avg_pool2d(x, x.size()[2:]).view(x.size(0), x.size(1))
        # predict transformation parameters
        theta = torch.tanh(self.fc(x))
        theta = theta.view(-1, 4, 2)
        return theta


class HarmAttn(nn.Module):
    """Harmonious Attention (Sec. 3.1)"""

    def __init__(self, in_channels):
        super(HarmAttn, self).__init__()
        self.soft_attn = SoftAttn(in_channels)
        # self.hard_attn = HardAttn(in_channels)

    def forward(self, x):
        y_soft_attn = self.soft_attn(x)
        # theta = self.hard_attn(x)
        return y_soft_attn#, theta  
        
        
        
        
        
        

def init_pretrained_weights(model, model_url):
    """Initializes model with pretrained weights.
    
    Layers that don't match with pretrained layers in name or size are kept unchanged.
    """
    pretrain_dict = model_zoo.load_url(model_url)
    model_dict = model.state_dict()
    pretrain_dict = {
        k: v
        for k, v in pretrain_dict.items()
        if k in model_dict and model_dict[k].size() == v.size()
    }
    model_dict.update(pretrain_dict)
    model.load_state_dict(model_dict)


"""ResNet"""


def resnet18(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = ResNet(
        num_classes=num_classes,
        loss=loss,
        block=BasicBlock,
        layers=[2, 2, 2, 2],
        last_stride=2,
        fc_dims=None,
        dropout_p=None,
        **kwargs
    )
    if pretrained:
        init_pretrained_weights(model, model_urls['resnet18'])
    return model


def resnet34(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = ResNet(
        num_classes=num_classes,
        loss=loss,
        block=BasicBlock,
        layers=[3, 4, 6, 3],
        last_stride=2,
        fc_dims=None,
        dropout_p=None,
        **kwargs
    )
    if pretrained:
        init_pretrained_weights(model, model_urls['resnet34'])
    return model


def resnet50(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = ResNet(
        num_classes=num_classes,
        loss=loss,
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        last_stride=2,
        fc_dims=[1024],
        dropout_p=None,
        **kwargs
    )
    if pretrained:
        init_pretrained_weights(model, model_urls['resnet50'])
    return model


def resnet101(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = ResNet(
        num_classes=num_classes,
        loss=loss,
        block=Bottleneck,
        layers=[3, 4, 23, 3],
        last_stride=2,
        fc_dims=None,
        dropout_p=None,
        **kwargs
    )
    if pretrained:
        init_pretrained_weights(model, model_urls['resnet101'])
    return model


def resnet152(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = ResNet(
        num_classes=num_classes,
        loss=loss,
        block=Bottleneck,
        layers=[3, 8, 36, 3],
        last_stride=2,
        fc_dims=None,
        dropout_p=None,
        **kwargs
    )
    if pretrained:
        init_pretrained_weights(model, model_urls['resnet152'])
    return model


"""ResNeXt"""


def resnext50_32x4d(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = ResNet(
        num_classes=num_classes,
        loss=loss,
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        last_stride=2,
        fc_dims=None,
        dropout_p=None,
        groups=32,
        width_per_group=4,
        **kwargs
    )
    if pretrained:
        init_pretrained_weights(model, model_urls['resnext50_32x4d'])
    return model


def resnext101_32x8d(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = ResNet(
        num_classes=num_classes,
        loss=loss,
        block=Bottleneck,
        layers=[3, 4, 23, 3],
        last_stride=2,
        fc_dims=None,
        dropout_p=None,
        groups=32,
        width_per_group=8,
        **kwargs
    )
    if pretrained:
        init_pretrained_weights(model, model_urls['resnext101_32x8d'])
    return model


"""
ResNet + FC
"""


def resnet50_fc512(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = ResNet(
        num_classes=num_classes,
        loss=loss,
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        last_stride=1,
        fc_dims=[512],
        dropout_p=None,
        **kwargs
    )
    if pretrained:
        init_pretrained_weights(model, model_urls['resnet50'])
    return model
# # from torchsummary import summary
# # model=resnet50_fc512(751)
# # summary(model,(3,224,64))






























# """
# Code source: https://github.com/pytorch/vision
# """
# from __future__ import division, absolute_import
# import torch.utils.model_zoo as model_zoo
# from torch import nn
# import math
# from torch.autograd import Function
# import torchvision.datasets as dsets
# from torchvision import transforms
# from torch.autograd import Variable
# import warnings
# import torch
# from torch import nn
# from torch.nn import functional as F
# # from .squeezenet import *
# import math




# __all__ = [
#     'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
#     'resnext50_32x4d', 'resnext101_32x8d', 'resnet50_fc512'
# ]

# model_urls = {
#     'resnet18':
#     'https://download.pytorch.org/models/resnet18-5c106cde.pth',
#     'resnet34':
#     'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
#     'resnet50':
#     'https://download.pytorch.org/models/resnet50-19c8e357.pth',
#     'resnet101':
#     'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
#     'resnet152':
#     'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
#     'resnext50_32x4d':
#     'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
#     'resnext101_32x8d':
#     'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
# }


# def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
#     """3x3 convolution with padding"""
#     return nn.Conv2d(
#         in_planes,
#         out_planes,
#         kernel_size=3,
#         stride=stride,
#         padding=dilation,
#         groups=groups,
#         bias=False,
#         dilation=dilation
#     )


# def conv1x1(in_planes, out_planes, stride=1):
#     """1x1 convolution"""
#     return nn.Conv2d(
#         in_planes, out_planes, kernel_size=1, stride=stride, bias=False
#     )


# class BasicBlock(nn.Module):
#     expansion = 1

#     def __init__(
#         self,
#         inplanes,
#         planes,
#         stride=1,
#         downsample=None,
#         groups=1,
#         base_width=64,
#         dilation=1,
#         norm_layer=None
#     ):
#         super(BasicBlock, self).__init__()
#         if norm_layer is None:
#             norm_layer = nn.BatchNorm2d
#         if groups != 1 or base_width != 64:
#             raise ValueError(
#                 'BasicBlock only supports groups=1 and base_width=64'
#             )
#         if dilation > 1:
#             raise NotImplementedError(
#                 "Dilation > 1 not supported in BasicBlock"
#             )
#         # Both self.conv1 and self.downsample layers downsample the input when stride != 1
#         self.conv1 = conv3x3(inplanes, planes, stride)
#         self.bn1 = norm_layer(planes)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = conv3x3(planes, planes)
#         self.bn2 = norm_layer(planes)
#         self.downsample = downsample
#         self.stride = stride

#     def forward(self, x):
#         identity = x

#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)

#         out = self.conv2(out)
#         out = self.bn2(out)

#         if self.downsample is not None:
#             identity = self.downsample(x)

#         out += identity
#         out = self.relu(out)

#         return out


# class Bottleneck(nn.Module):
#     expansion = 4

#     def __init__(
#         self,
#         inplanes,
#         planes,
#         stride=1,
#         downsample=None,
#         groups=1,
#         base_width=64,
#         dilation=1,
#         norm_layer=None
#     ):
#         super(Bottleneck, self).__init__()
#         if norm_layer is None:
#             norm_layer = nn.BatchNorm2d
#         width = int(planes * (base_width/64.)) * groups
#         # Both self.conv2 and self.downsample layers downsample the input when stride != 1
#         self.conv1 = conv1x1(inplanes, width)
#         self.bn1 = norm_layer(width)
#         self.conv2 = conv3x3(width, width, stride, groups, dilation)
#         self.bn2 = norm_layer(width)
#         self.conv3 = conv1x1(width, planes * self.expansion)
#         self.bn3 = norm_layer(planes * self.expansion)
#         self.relu = nn.ReLU(inplace=True)
#         self.downsample = downsample
#         self.stride = stride

#     def forward(self, x):
#         identity = x

#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)

#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.relu(out)

#         out = self.conv3(out)
#         out = self.bn3(out)

#         if self.downsample is not None:
#             identity = self.downsample(x)

#         out += identity
#         out = self.relu(out)

#         return out


# class ResNet(nn.Module):
#     """Residual network.
    
#     Reference:
#         - He et al. Deep Residual Learning for Image Recognition. CVPR 2016.
#         - Xie et al. Aggregated Residual Transformations for Deep Neural Networks. CVPR 2017.

#     Public keys:
#         - ``resnet18``: ResNet18.
#         - ``resnet34``: ResNet34.
#         - ``resnet50``: ResNet50.
#         - ``resnet101``: ResNet101.
#         - ``resnet152``: ResNet152.
#         - ``resnext50_32x4d``: ResNeXt50.
#         - ``resnext101_32x8d``: ResNeXt101.
#         - ``resnet50_fc512``: ResNet50 + FC.
#     """

#     def __init__(
#         self,
#         num_classes,
#         loss,
#         block,
#         layers,
#         zero_init_residual=False,
#         groups=1,
#         width_per_group=64,
#         replace_stride_with_dilation=None,
#         norm_layer=None,
#         last_stride=2,
#         fc_dims=None,
#         dropout_p=None,
#         **kwargs
#     ):
#         super(ResNet, self).__init__()
#         if norm_layer is None:
#             norm_layer = nn.BatchNorm2d
#         self._norm_layer = norm_layer
#         self.loss = loss
#         self.feature_dim = 512 * block.expansion
#         self.inplanes = 64
#         self.dilation = 1
#         if replace_stride_with_dilation is None:
#             # each element in the tuple indicates if we should replace
#             # the 2x2 stride with a dilated convolution instead
#             replace_stride_with_dilation = [False, False, False]
#         if len(replace_stride_with_dilation) != 3:
#             raise ValueError(
#                 "replace_stride_with_dilation should be None "
#                 "or a 3-element tuple, got {}".
#                 format(replace_stride_with_dilation)
#             )
#         self.groups = groups
#         self.base_width = width_per_group
#         self.conv1 = nn.Conv2d(
#             3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False
#         )
#         self.bn1 = norm_layer(self.inplanes)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         self.layer1 = self._make_layer(block, 64, layers[0])
#         self.layer2 = self._make_layer(
#             block,
#             128,
#             layers[1],
#             stride=2,
#             dilate=replace_stride_with_dilation[0]
#         )
#         self.layer3 = self._make_layer(
#             block,
#             256,
#             layers[2],
#             stride=2,
#             dilate=replace_stride_with_dilation[1]
#         )
#         self.layer4 = self._make_layer(
#             block,
#             512,
#             layers[3],
#             stride=last_stride,
#             dilate=replace_stride_with_dilation[2]
#         )
#         self.global_avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc = self._construct_fc_layer(
#             fc_dims, 512 * block.expansion, dropout_p
#         )
#         self.classifier = nn.Linear(128, num_classes)
        
        
    
#         self._init_params()


#         self.fc_encode = nn.Linear(self.feature_dim, 128)



        
#         # Zero-initialize the last BN in each residual branch,
#         # so that the residual branch starts with zeros, and each residual block behaves like an identity.
#         # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
#         if zero_init_residual:
#             for m in self.modules():
#                 if isinstance(m, Bottleneck):
#                     nn.init.constant_(m.bn3.weight, 0)
#                 elif isinstance(m, BasicBlock):
#                     nn.init.constant_(m.bn2.weight, 0)

#     def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
#         norm_layer = self._norm_layer
#         downsample = None
#         previous_dilation = self.dilation
#         if dilate:
#             self.dilation *= stride
#             stride = 1
#         if stride != 1 or self.inplanes != planes * block.expansion:
#             downsample = nn.Sequential(
#                 conv1x1(self.inplanes, planes * block.expansion, stride),
#                 norm_layer(planes * block.expansion),
#             )

#         layers = []
#         layers.append(
#             block(
#                 self.inplanes, planes, stride, downsample, self.groups,
#                 self.base_width, previous_dilation, norm_layer
#             )
#         )
#         self.inplanes = planes * block.expansion
#         for _ in range(1, blocks):
#             layers.append(
#                 block(
#                     self.inplanes,
#                     planes,
#                     groups=self.groups,
#                     base_width=self.base_width,
#                     dilation=self.dilation,
#                     norm_layer=norm_layer
#                 )
#             )

#         return nn.Sequential(*layers)

#     def _construct_fc_layer(self, fc_dims, input_dim, dropout_p=None):
#         """Constructs fully connected layer

#         Args:
#             fc_dims (list or tuple): dimensions of fc layers, if None, no fc layers are constructed
#             input_dim (int): input dimension
#             dropout_p (float): dropout probability, if None, dropout is unused
#         """
#         if fc_dims is None:
#             self.feature_dim = input_dim
#             return None

#         assert isinstance(
#             fc_dims, (list, tuple)
#         ), 'fc_dims must be either list or tuple, but got {}'.format(
#             type(fc_dims)
#         )

#         layers = []
#         for dim in fc_dims:
#             layers.append(nn.Linear(input_dim, dim))
#             layers.append(nn.BatchNorm1d(dim))
#             layers.append(nn.ReLU(inplace=True))
#             if dropout_p is not None:
#                 layers.append(nn.Dropout(p=dropout_p))
#             input_dim = dim

#         self.feature_dim = fc_dims[-1]

#         return nn.Sequential(*layers)

#     def _init_params(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(
#                     m.weight, mode='fan_out', nonlinearity='relu'
#                 )
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm1d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 nn.init.normal_(m.weight, 0, 0.01)
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)

#     def featuremaps(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
#         return x

#     def forward(self, x):
#         f = self.featuremaps(x)

#         v = self.global_avgpool(f)
#         v = v.view(v.size(0), -1)
#         if self.fc is not None:
#             v = self.fc(v)





#         h=self.fc_encode(v)
#         y = self.classifier(h)
        
        
        
        
        
        
        
        
#         b=hash_layer(h)
#         if self.loss == 'softmax':
#             return y
#         elif self.loss == 'triplet':
#             return y, v,h,b#,b_weighted#,y_pcb,v_pcb_return
#         else:
#             raise KeyError("Unsupported loss: {}".format(self.loss))












# import numpy as np
# # new layer
# class hash(Function):
#     @staticmethod
#     def forward(ctx, input):
        
#         # ctx.save_for_backward(input)
#         # print(output)
#         # return output
        
#         # output=np.array(input.shape)
        
        
        
        
#         # for i in range(input.shape[0]):
#         #     for j in range(input.shape[1]):
#         #         if input[i][j]>=input_average[j]:
#         #             output[i][j]=1
#         #         else:
#         #             output[i][j]=-1
#         # print(input)
#         # print(output)
#         # print(torch.sign(input))
#         # output=torch.sign(input)
#         # print(output.dtype)
#         return torch.sign(input)

#     @staticmethod
#     def backward(ctx, grad_output):
#         # input,  = ctx.saved_tensors
#         # grad_output = grad_output.data

#         return grad_output


# def hash_layer(input):
#     return hash.apply(input)  








        
        
        
        
        
        
        

# def init_pretrained_weights(model, model_url):
#     """Initializes model with pretrained weights.
    
#     Layers that don't match with pretrained layers in name or size are kept unchanged.
#     """
#     pretrain_dict = model_zoo.load_url(model_url)
#     model_dict = model.state_dict()
#     pretrain_dict = {
#         k: v
#         for k, v in pretrain_dict.items()
#         if k in model_dict and model_dict[k].size() == v.size()
#     }
#     model_dict.update(pretrain_dict)
#     model.load_state_dict(model_dict)


# """ResNet"""


# def resnet18(num_classes, loss='softmax', pretrained=True, **kwargs):
#     model = ResNet(
#         num_classes=num_classes,
#         loss=loss,
#         block=BasicBlock,
#         layers=[2, 2, 2, 2],
#         last_stride=2,
#         fc_dims=None,
#         dropout_p=None,
#         **kwargs
#     )
#     if pretrained:
#         init_pretrained_weights(model, model_urls['resnet18'])
#     return model


# def resnet34(num_classes, loss='softmax', pretrained=True, **kwargs):
#     model = ResNet(
#         num_classes=num_classes,
#         loss=loss,
#         block=BasicBlock,
#         layers=[3, 4, 6, 3],
#         last_stride=2,
#         fc_dims=None,
#         dropout_p=None,
#         **kwargs
#     )
#     if pretrained:
#         init_pretrained_weights(model, model_urls['resnet34'])
#     return model


# def resnet50(num_classes, loss='softmax', pretrained=True, **kwargs):
#     model = ResNet(
#         num_classes=num_classes,
#         loss=loss,
#         block=Bottleneck,
#         layers=[3, 4, 6, 3],
#         last_stride=2,
#         fc_dims=None,
#         dropout_p=None,
#         **kwargs
#     )
#     if pretrained:
#         init_pretrained_weights(model, model_urls['resnet50'])
#     return model


# def resnet101(num_classes, loss='softmax', pretrained=True, **kwargs):
#     model = ResNet(
#         num_classes=num_classes,
#         loss=loss,
#         block=Bottleneck,
#         layers=[3, 4, 23, 3],
#         last_stride=2,
#         fc_dims=None,
#         dropout_p=None,
#         **kwargs
#     )
#     if pretrained:
#         init_pretrained_weights(model, model_urls['resnet101'])
#     return model


# def resnet152(num_classes, loss='softmax', pretrained=True, **kwargs):
#     model = ResNet(
#         num_classes=num_classes,
#         loss=loss,
#         block=Bottleneck,
#         layers=[3, 8, 36, 3],
#         last_stride=2,
#         fc_dims=None,
#         dropout_p=None,
#         **kwargs
#     )
#     if pretrained:
#         init_pretrained_weights(model, model_urls['resnet152'])
#     return model


# """ResNeXt"""


# def resnext50_32x4d(num_classes, loss='softmax', pretrained=True, **kwargs):
#     model = ResNet(
#         num_classes=num_classes,
#         loss=loss,
#         block=Bottleneck,
#         layers=[3, 4, 6, 3],
#         last_stride=2,
#         fc_dims=None,
#         dropout_p=None,
#         groups=32,
#         width_per_group=4,
#         **kwargs
#     )
#     if pretrained:
#         init_pretrained_weights(model, model_urls['resnext50_32x4d'])
#     return model


# def resnext101_32x8d(num_classes, loss='softmax', pretrained=True, **kwargs):
#     model = ResNet(
#         num_classes=num_classes,
#         loss=loss,
#         block=Bottleneck,
#         layers=[3, 4, 23, 3],
#         last_stride=2,
#         fc_dims=None,
#         dropout_p=None,
#         groups=32,
#         width_per_group=8,
#         **kwargs
#     )
#     if pretrained:
#         init_pretrained_weights(model, model_urls['resnext101_32x8d'])
#     return model


# """
# ResNet + FC
# """


# def resnet50_fc512(num_classes, loss='softmax', pretrained=True, **kwargs):
#     model = ResNet(
#         num_classes=num_classes,
#         loss=loss,
#         block=Bottleneck,
#         layers=[3, 4, 6, 3],
#         last_stride=1,
#         fc_dims=[512],
#         dropout_p=None,
#         **kwargs
#     )
#     if pretrained:
#         init_pretrained_weights(model, model_urls['resnet50'])
#     return model
# # from torchsummary import summary
# # model=resnet50_fc512(751)
# # summary(model,(3,224,64))

# from torchsummary import summary
# model=resnet50_fc512(751)
# summary(model.cuda(),(3,384,128))