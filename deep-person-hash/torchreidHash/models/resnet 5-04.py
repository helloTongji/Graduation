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
from torch.nn import init
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



class Bottleneck_pcb(nn.Module):
    expansion = 2

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
        super(Bottleneck_pcb, self).__init__()
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
        # print(last_stride)
        self.layer4 = self._make_layer(
            block,
            512,
            layers[3],
            stride=last_stride,
            dilate=replace_stride_with_dilation[2]
        )
        # self.global_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = self._construct_fc_layer(
        #     fc_dims, 512 * block.expansion, dropout_p
        # )
        
        
        
        
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        assert fc_dims is not None
        self.fc_fusion = self._construct_fc_layer(
            fc_dims, 512 * block.expansion * 2
        )
        self.feature_dim += 512 * block.expansion
        self.classifier = nn.Linear(self.hashbit, num_classes)
        self.classifier2 = nn.Linear(self.hashbit, num_classes)
        #self._init_params()

        
        
        
        # if(self.hashbit<=128):
        #     self.fc_encode = nn.Linear(self.feature_dim, self.hashbit)
        #     self.classifier = nn.Linear(self.hashbit, num_classes)
        # else:
        #     self.fc_encode = nn.Linear(self.feature_dim, self.hashbit//2)
        #     self.classifier = nn.Linear(self.hashbit//2, num_classes)
        # self.fc_encode = nn.Linear(self.feature_dim, 128)
        
        
        
        

        
        
        
        
        
        
        self.layer4_pcb = self._make_layer_pcb(block, 512, layers[3], stride=1)
        num_stripes=6
        local_conv_out_channels=256
        self.layer_nums = 3 
        self.num_stripes = 6
    
        self.local_conv_list = nn.ModuleList()
        for _ in range(3*self.num_stripes-3):
          self.local_conv_list.append(nn.Sequential(
            nn.Conv2d(2048, local_conv_out_channels, 1),
            nn.BatchNorm2d(local_conv_out_channels),
            nn.ReLU(inplace=True)
          ))

        if num_classes > 0:
            self.fc_list = nn.ModuleList()
            for _ in range(3*num_stripes-3):
                fc = nn.Linear(local_conv_out_channels, num_classes)
                init.normal(fc.weight, std=0.001)
                init.constant(fc.bias, 0)
                self.fc_list.append(fc)
        

        # if(self.hashbit<=128):
        #     self.fc_encode_pcb = nn.Linear(local_conv_out_channels*15, self.hashbit)
        # else:
        #     self.fc_encode_pcb = nn.Linear(local_conv_out_channels*15, self.hashbit-self.hashbit//2)
        self.split_num=10
        self.fc_split = nn.Linear(local_conv_out_channels*15+self.feature_dim, self.hashbit*self.split_num)
        
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
    def _make_layer_pcb(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
          downsample = nn.Sequential(
            nn.Conv2d(self.inplanes, planes * block.expansion,
                      kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(planes * block.expansion),
          )
    
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
          layers.append(block(self.inplanes, planes))
    
        return nn.Sequential(*layers)
    
    
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
    
    def _make_layer_pcb(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        inplanes=1024
        # print(planes * block.expansion)
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )
        
        layers = []
        layers.append(
            block(
                inplanes, planes, stride, downsample, self.groups,
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
        x4a = self.layer4[0](x)
        x4b = self.layer4[1](x4a)
        x4c = self.layer4[2](x4b)
        return x4a, x4b, x4c
    def featuremaps_pcb(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4_pcb(x)
        return x
    


    
    def forward(self, x):
        
        x4a, x4b, x4c = self.featuremaps(x)

        v4a = self.global_avgpool(x4a)
        v4b = self.global_avgpool(x4b)
        v4c = self.global_avgpool(x4c)
        v4ab = torch.cat([v4a, v4b], 1)
        v4ab = v4ab.view(v4ab.size(0), -1)
        v4ab = self.fc_fusion(v4ab)
        v4c = v4c.view(v4c.size(0), -1)
        v = torch.cat([v4ab, v4c], 1)
        # h=self.fc_encode(v)
        # y = self.classifier(h)
        # b=hash_layer(h)
        
        
        
        
        
        
        feat = self.featuremaps_pcb(x)
        assert feat.size(2) % self.num_stripes == 0
        stripe_h = int(feat.size(2) / self.num_stripes)
        local_feat_list = []
        logits_list = [] 
        self.layer_nums=3
        if self.layer_nums >= 1:
            for i in range(self.num_stripes):
    	    # shape [N, C, 1, 1]
                local_feat = F.avg_pool2d(
                feat[:, :, i * stripe_h: (i + 1) * stripe_h, :],
                (stripe_h, feat.size(-1)))
                # shape [N, c, 1, 1]
                local_feat = self.local_conv_list[i](local_feat)
                # shape [N, c]
                local_feat = local_feat.view(local_feat.size(0), -1)
                local_feat_list.append(local_feat)
                if hasattr(self, 'fc_list'):
                  logits_list.append(self.fc_list[i](local_feat))
    
        if self.layer_nums >= 2:
            for i in range(self.num_stripes-1):
                # shape [N, C, 1, 1]
                local_feat = F.avg_pool2d(
                feat[:, :, i * stripe_h: (i + 2) * stripe_h, :],
                (2*stripe_h, feat.size(-1)))
                # shape [N, c, 1, 1]
                local_feat = self.local_conv_list[self.num_stripes+i](local_feat)
                # shape [N, c]
                local_feat = local_feat.view(local_feat.size(0), -1)
                local_feat_list.append(local_feat)
                if hasattr(self, 'fc_list'):
                    logits_list.append(self.fc_list[self.num_stripes+i](local_feat))
    
        if self.layer_nums >= 3:
            for i in range(self.num_stripes-2):
                # shape [N, C, 1, 1]
                local_feat = F.avg_pool2d(
                feat[:, :, i * stripe_h: (i + 3) * stripe_h, :],
                (3*stripe_h, feat.size(-1)))
                # shape [N, c, 1, 1]
                local_feat = self.local_conv_list[2*self.num_stripes-1+i](local_feat)
                # shape [N, c]
                local_feat = local_feat.view(local_feat.size(0), -1)
                local_feat_list.append(local_feat)
                if hasattr(self, 'fc_list'):
                    logits_list.append(self.fc_list[2*self.num_stripes-1+i](local_feat))

        if hasattr(self, 'fc_list'):
            feat_pcb=torch.cat([local_feat_list[0],local_feat_list[1],local_feat_list[2],local_feat_list[3],
            local_feat_list[4],local_feat_list[5],local_feat_list[6],local_feat_list[7],local_feat_list[8],
            local_feat_list[9],local_feat_list[10],local_feat_list[11],local_feat_list[12],local_feat_list[13]
            ,local_feat_list[14]],dim=1)
            # logits_pcb=torch.cat([logits_list[0],logits_list[1],logits_list[2],logits_list[3],
            # logits_list[4],logits_list[5],logits_list[6],logits_list[7],logits_list[8],
            # logits_list[9],logits_list[10],logits_list[11],logits_list[12],logits_list[13]
            # ,logits_list[14]],dim=1)
            
        
        
        # h_pcb=self.fc_encode_pcb(feat_pcb)
        # b_pcb=hash_layer(h_pcb)
        
        # if(self.hashbit<=128):
        #     b_return=b
        #     h_return=h
        # else:
        #     h_return=torch.cat([h,h_pcb],dim=1)
        #     b_return=torch.cat([b,b_pcb],dim=1)
        v=torch.cat([v,feat_pcb],dim=1)
        v_split=self.fc_split(v)
        h_return=self.de1(v_split)
        y = self.classifier(h_return)
        b_return=hash_layer(h_return)
        b_return_classify=self.classifier2(b_return)
        #y=torch.cat([y,logits_pcb],dim=1)
        
        if self.loss == 'softmax':
            return y
        elif self.loss == 'triplet':
            return y, v,h_return,b_return,logits_list,b_return_classify#,fg_p3#,b_weighted#,y_pcb,v_pcb_return
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

class DimReduceLayer(nn.Module):

    def __init__(self, in_channels, out_channels, nonlinear):
        super(DimReduceLayer, self).__init__()
        layers = []
        layers.append(
            nn.Conv2d(
                in_channels, out_channels, 1, stride=1, padding=0, bias=False
            )
        )
        layers.append(nn.BatchNorm2d(out_channels))

        if nonlinear == 'relu':
            layers.append(nn.ReLU(inplace=True))
        elif nonlinear == 'leakyrelu':
            layers.append(nn.LeakyReLU(0.1))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
import numpy as np
# new layer
class hash(Function):
    @staticmethod
    def forward(ctx, input):
        
        # ctx.save_for_backward(input)
        # print(output)
        # return output
        
        # output=np.array(input.shape)
        
        
        
        
        # for i in range(input.shape[0]):
        #     for j in range(input.shape[1]):
        #         if input[i][j]>=input_average[j]:
        #             output[i][j]=1
        #         else:
        #             output[i][j]=-1
        # print(input)
        # print(output)
        # print(torch.sign(input))
        # output=torch.sign(input)
        # print(output.dtype)
        return torch.sign(input)

    @staticmethod
    def backward(ctx, grad_output):
        # input,  = ctx.saved_tensors
        # grad_output = grad_output.data

        return grad_output


def hash_layer(input):
    return hash.apply(input)  




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

#from torchsummary import summary
#model=resnet50_fc512(751)
#summary(model,(3,384,128))




























class PCBHighDivModule(nn.Module):
    def __init__(self, in_channels, order=1):
        super(PCBHighDivModule, self).__init__()
        self.order = order
        self.inter_channels = in_channels // 8 * 2
        for j in range(self.order):
            for i in range(j+1):
                name = 'order' + str(self.order) + '_' + str(j+1) + '_' + str(i+1)
                setattr(self, name, nn.Sequential(nn.Conv2d(in_channels, self.inter_channels, 1, padding=0, bias=False))
            )
        for i in range(self.order):
            name = 'convb' + str(self.order) + '_' + str(i+1)
            setattr(self, name, nn.Sequential(nn.Conv2d(self.inter_channels, in_channels, 1, padding=0, bias=False),
                                              nn.Sigmoid()
                                   )
                                   )

    def forward(self, x):
        y=[]
        for j in range(self.order):
            for i in range(j+1):
                name = 'order' + str(self.order) + '_' + str(j+1) + '_' + str(i+1)
                layer = getattr(self, name)
                y.append(layer(x))
        y_ = []
        cnt=0
        for j in range(self.order):
            y_temp = 1
            for i in range(j+1):
                y_temp = y_temp * y[cnt]
                cnt += 1
            y_.append(F.relu(y_temp))
        
        #y_ = F.relu(y_)
        y__ = 0
        for i in range(self.order):
            name = 'convb' + str(self.order) + '_' + str(i+1)
            layer = getattr(self, name)
            y__ += layer(y_[i])
        out = x * y__ / self.order
        return out

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
