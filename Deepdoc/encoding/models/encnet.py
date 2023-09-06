###########################################################################
# Created by: Hang Zhang 
# Email: zhang.hang@rutgers.edu 
# Copyright (c) 2017
###########################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
import Deepdoc.encoding as encoding

import os

from .base import BaseNet
#from base import BaseNet
from .fcn import FCNHead

__all__ = ['EncNet', 'EncModule', 'get_encnet', 'get_encnet_resnet50_pcontext',
           'get_encnet_resnet101_pcontext', 'get_encnet_resnet50_ade',
           'get_encnet_resnet101_ade']

import numpy as np

class MobileNetConv2d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MobileNetConv2d, self).__init__()
        
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x
    
class EncNet(BaseNet):
    def __init__(self, nclass, backbone, aux=True, se_loss=True,
                 norm_layer=nn.BatchNorm2d, **kwargs):
        super(EncNet, self).__init__(nclass, backbone, aux, se_loss,
                                     norm_layer=norm_layer, **kwargs)
        
        #self.feat_channel=1024 if backbone=='densenet' else 2048
        #self.w=32
        self.w=40   # -1 layer or hyper parameter
        #self.head = EncHead([48,24, self.feat_channel], self.nclass, se_loss=se_loss, jpu=kwargs['jpu'],
        #                    lateral=kwargs['lateral'], norm_layer=norm_layer,
        #                    up_kwargs=self._up_kwargs)
        #densenet169
        #self.head = EncHead([512, 1280, self.feat_channel], self.nclass, se_loss=se_loss, jpu=kwargs['jpu'],
        #                    lateral=kwargs['lateral'], norm_layer=norm_layer,
        #                    up_kwargs=self._up_kwargs)
        
        #                   -4  -3  -1 layer
        self.head = EncHead([48,32,40], self.nclass, se_loss=se_loss, jpu=kwargs['jpu'],
                                              lateral=kwargs['lateral'], norm_layer=norm_layer,
                                              up_kwargs=self._up_kwargs)
        
        if aux:

            #                       -3 layer
            self.auxlayer = FCNHead(32, nclass, norm_layer=norm_layer)
            
            #self.auxlayer = FCNHead(24, nclass, norm_layer=norm_layer)
            #self.auxlayer = FCNHead(1280, nclass, norm_layer=norm_layer)
        #self._se_reduce = nn.Conv2d(
        #    in_channels=self.w*3, out_channels=self.w, kernel_size=1)
        #self._se_expand = nn.Conv2d(
        #    in_channels=self.w, out_channels=self.w*3, kernel_size=1)
        #self.act=nn.ReLU(inplace=True)
        # 4=256 5=512 6=1024 
        
        #          edge_4 -> -4 layer     edge_5 -> -3 layer    edge_6 -> -1 layer
        '''3/29
        self.edge_4 = nn.Sequential(nn.Conv2d(96,  self.w, 3, padding=1, bias=False),
                                   norm_layer(self.w),
                                   nn.ReLU(inplace=True))
        self.edge_5 = nn.Sequential(nn.Conv2d(32, self.w, 3, padding=1, bias=False),
                                   norm_layer(self.w),
                                   nn.ReLU(inplace=True))
        self.edge_6 = nn.Sequential(nn.Conv2d(40, self.w, 3, padding=1, bias=False),
                                   norm_layer(self.w),
                                   nn.ReLU(inplace=True))
        '''
        self.edge_4 = nn.Sequential(MobileNetConv2d(96,self.w),
                                   norm_layer(self.w),
                                   nn.ReLU(inplace=True))
        self.edge_5 = nn.Sequential(MobileNetConv2d(32,self.w),
                                   norm_layer(self.w),
                                   nn.ReLU(inplace=True))
        self.edge_6 = nn.Sequential(MobileNetConv2d(40,self.w),
                                   norm_layer(self.w),
                                   nn.ReLU(inplace=True))
        self.c = nn.Conv2d(self.w*3, 1, 1, bias=False)
        # self.edge_7 = nn.Sequential(nn.Conv2d(48, self.w, 3, padding=1, bias=False),
        #                            norm_layer(self.w),
        #                            nn.ReLU(inplace=True))
        # self.edge_8 = nn.Sequential(nn.Conv2d(96, self.w, 3, padding=1, bias=False),
        #                            norm_layer(self.w),
        #                            nn.ReLU(inplace=True))
                       
        #densenet169
        #self.edge_6 = nn.Sequential(nn.Conv2d(1280, self.w, 3, padding=1, bias=False),
        #                            norm_layer(self.w),
        #                            nn.ReLU(inplace=True))
        #self.edge_7 = nn.Sequential(nn.Conv2d(1024, self.w, 3, padding=1, bias=False),
        #                           norm_layer(self.w),
        #                           nn.ReLU(inplace=True))
        #self.c=nn.Conv2d(self.w*4,1,1,bias=False)
        '''
        self.c = nn.Conv2d(self.w*3, 1, 1, bias=False)
        '''
    def forward(self, x):
        imsize = x.size()[2:]
        #print('OK2')
        features = self.base_forward(x)#features[-1]=[8,2048,60,60]
        #print(features)
        #features[-1]=self.fpa(features[-1])
        x = list(self.head(*features))#x[0]=[8,512,60,60]
        # print(a[0].shape,a[1].shape)

        x[0] = F.upsample(x[0], imsize, **self._up_kwargs)#imsize [480,480]
        
        if self.aux:
            auxout = self.auxlayer(features[-4])
            auxout = F.upsample(auxout, imsize, **self._up_kwargs)
            x.append(auxout)
        
        y = F.interpolate(self.edge_4(features[1]),imsize,**self._up_kwargs)
        y1 = F.interpolate(self.edge_5(features[2]),imsize,**self._up_kwargs)
        y2 = F.interpolate(self.edge_6(features[3]),imsize,**self._up_kwargs)
        # y3 = F.interpolate(self.edge_7(features[-4]),imsize,**self._up_kwargs)
        # y4 = F.interpolate(self.edge_8(features[-5]),imsize,**self._up_kwargs)
        
        # yres = self.c(torch.cat([y, y1, y2,y3,y4], 1))
        yres = self.c(torch.cat([y, y1, y2], 1))
        #yres = torch.cat([y, y1, y2], 1)
        #x_squeezed = F.adaptive_avg_pool2d(yres, 1)
        #x_squeezed = self._se_expand(
        #    self.act(self._se_reduce(x_squeezed)))
        #yres = torch.sigmoid(x_squeezed) * yres
        #yres=self.c(yres)
        x.append(yres)
        
        return tuple(x)


class EncModule(nn.Module):
    def __init__(self, in_channels, nclass, ncodes=32, se_loss=True, norm_layer=None):
        super(EncModule, self).__init__()
        self.se_loss = se_loss
        self.encoding = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            norm_layer(in_channels),
            nn.ReLU(inplace=True),
            encoding.nn.Encoding(D=in_channels, K=ncodes),
            encoding.nn.BatchNorm1d(ncodes),
            nn.ReLU(inplace=True),
            encoding.nn.Mean(dim=1))
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.Sigmoid())
        if self.se_loss:
            #self.selayer_2 = nn.Linear(in_channels, nclass)
            self.selayer = nn.Linear(in_channels, nclass)

    def forward(self, x):
        en = self.encoding(x)
        b, c, _, _ = x.size()
        gamma = self.fc(en)
        y = gamma.view(b, c, 1, 1)
        outputs = [F.relu_(x + x * y)]

        if self.se_loss:
            outputs.append(self.selayer(en))
        return tuple(outputs)


class EncHead(nn.Module):
    #*
    def __init__(self, in_channels, out_channels, se_loss=True, jpu=True, lateral=False,
                 norm_layer=None, up_kwargs=None):
        super(EncHead, self).__init__()
        self.se_loss = se_loss
        self.lateral = lateral
        self.up_kwargs = up_kwargs
        # inital 512
        self.conv5 = nn.Sequential(nn.Conv2d(in_channels[-1], 512, 1, bias=False),
                                   norm_layer(512),
                                   nn.ReLU(inplace=True)) if jpu else \
                     nn.Sequential(nn.Conv2d(in_channels[-1], 512, 3, padding=1, bias=False),
                                   norm_layer(512),
                                   nn.ReLU(inplace=True))
        if lateral:
            self.connect = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(in_channels[0], 512, kernel_size=1, bias=False),
                    norm_layer(512),
                    nn.ReLU(inplace=True)),
                nn.Sequential(
                    nn.Conv2d(in_channels[1], 512, kernel_size=1, bias=False),
                    norm_layer(512),
                    nn.ReLU(inplace=True)),
            ])
            self.fusion = nn.Sequential(
                    nn.Conv2d(3*512, 512, kernel_size=3, padding=1, bias=False),
                    norm_layer(512),
                    nn.ReLU(inplace=True))
        self.encmodule = EncModule(512, out_channels, ncodes=32,
            se_loss=se_loss, norm_layer=norm_layer)
        self.conv6 = nn.Sequential(nn.Dropout2d(0.1, False),
                                   nn.Conv2d(512, out_channels, 1))
        #self.conv6_2 = nn.Sequential(nn.Dropout2d(0.1, False),
        #                           nn.Conv2d(512, out_channels, 1))

    def forward(self, *inputs):#2048=>512
        feat = self.conv5(inputs[-1])
        if self.lateral:
            c2 = self.connect[0](inputs[1])
            c3 = self.connect[1](inputs[2])
            c3 = F.upsample(c3, (c2.shape[2], c2.shape[3]), **self.up_kwargs)
            feat = self.fusion(torch.cat([feat, c2, c3], 1))

        outs = list(self.encmodule(feat))
        outs[0] = self.conv6(outs[0])
        return tuple(outs)


def get_encnet(dataset='pascal_voc', backbone='resnet50', pretrained=False,
               root='/data/4TB/richard/richard/DocNet/encoding/architecture', **kwargs):
    r"""EncNet model from the paper `"Context Encoding for Semantic Segmentation"
    <https://arxiv.org/pdf/1803.08904.pdf>`_

    Parameters
    ----------
    dataset : str, default pascal_voc
        The dataset that model pretrained on. (pascal_voc, ade20k)
    backbone : str, default resnet50
        The backbone network. (resnet50, 101, 152)
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.encoding/models'
        Location for keeping the model parameters.

    
    Examples
    --------
    >>> model = get_encnet(dataset='pascal_voc', backbone='resnet50', pretrained=False)
    >>> print(model)
    """
    print(root)
    acronyms = {
        'pascal_voc': 'voc',
        'ade20k': 'ade',
        'pcontext': 'pcontext',
    }
    # infer number of classes
    from ..datasets import datasets
    model = EncNet(8, backbone=backbone, root=root, **kwargs)
    
    if pretrained:
        from .model_store import get_model_file
        model.load_state_dict(torch.load(
            get_model_file('encnet_%s_%s'%(backbone, acronyms[dataset]), root=root)))
    return model


def get_encnet_resnet50_pcontext(pretrained=False, root=os.path.abspath(os.path.join(os.getcwd(), '..')), **kwargs):
    r"""EncNet-PSP model from the paper `"Context Encoding for Semantic Segmentation"
    <https://arxiv.org/pdf/1803.08904.pdf>`_

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.encoding/models'
        Location for keeping the model parameters.


    Examples
    --------
    >>> model = get_encnet_resnet50_pcontext(pretrained=True)
    >>> print(model)
    """
    return get_encnet('pcontext', 'resnet50', pretrained, root=root, aux=True,
                      base_size=520, crop_size=480, **kwargs)

def get_encnet_resnet101_pcontext(pretrained=False, root='~/.encoding/models', **kwargs):
    r"""EncNet-PSP model from the paper `"Context Encoding for Semantic Segmentation"
    <https://arxiv.org/pdf/1803.08904.pdf>`_

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.encoding/models'
        Location for keeping the model parameters.


    Examples
    --------
    >>> model = get_encnet_resnet101_pcontext(pretrained=True)
    >>> print(model)
    """
    return get_encnet('pcontext', 'resnet101', pretrained, root=root, aux=True,
                      base_size=520, crop_size=480, lateral=True, **kwargs)

def get_encnet_resnet50_ade(pretrained=False, root='~/.encoding/models', **kwargs):
    r"""EncNet-PSP model from the paper `"Context Encoding for Semantic Segmentation"
    <https://arxiv.org/pdf/1803.08904.pdf>`_

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.encoding/models'
        Location for keeping the model parameters.


    Examples
    --------
    >>> model = get_encnet_resnet50_ade(pretrained=True)
    >>> print(model)
    """
    return get_encnet('ade20k', 'resnet50', pretrained, root=root, aux=True,
                      base_size=520, crop_size=480, **kwargs)

def get_encnet_resnet101_ade(pretrained=False, root='~/.encoding/models', **kwargs):
    r"""EncNet-PSP model from the paper `"Context Encoding for Semantic Segmentation"
    <https://arxiv.org/pdf/1803.08904.pdf>`_

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.encoding/models'
        Location for keeping the model parameters.


    Examples
    --------
    >>> model = get_encnet_resnet50_ade(pretrained=True)
    >>> print(model)
    """
    return get_encnet('ade20k', 'resnet101', pretrained, root=root, aux=True,
                      base_size=640, crop_size=576, lateral=True, **kwargs)

def get_encnet_resnet152_ade(pretrained=False, root='~/.encoding/models', **kwargs):
    r"""EncNet-PSP model from the paper `"Context Encoding for Semantic Segmentation"
    <https://arxiv.org/pdf/1803.08904.pdf>`_

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.encoding/models'
        Location for keeping the model parameters.


    Examples
    --------
    >>> model = get_encnet_resnet50_ade(pretrained=True)
    >>> print(model)
    """
    return get_encnet('ade20k', 'resnet152', pretrained, root=root, aux=True,
                      base_size=520, crop_size=480, **kwargs)

if __name__ == '__main__':
       print(123)