##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Hang Zhang
## ECE Department, Rutgers University
## Email: zhang.hang@rutgers.edu
## Copyright (c) 2017
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

"""Encoding Custermized NN Module"""
import torch
import torch.nn as nn

from torch.nn import functional as F
from torch.nn import Module, Sequential, Conv2d, ReLU, AdaptiveAvgPool2d, BCELoss, CrossEntropyLoss

from torch.autograd import Variable
import numpy as np
from  torchvision.utils import save_image
torch_ver = torch.__version__[:3]

__all__ = ['SegmentationLosses', 'PyramidPooling', 'Mean']

class SegmentationLosses(CrossEntropyLoss):
    """2D Cross Entropy Loss with Auxilary Loss"""
    def __init__(self, se_loss=False, se_weight=0.2, nclass=-1,
                 aux=False, aux_weight=0.4, weight=None,
                 size_average=True, ignore_index=-1):
        super(SegmentationLosses, self).__init__(weight, size_average, ignore_index)
        self.se_loss = se_loss
        self.aux = aux
        self.nclass = nclass
        self.se_weight = se_weight
        self.aux_weight = aux_weight
        self.bceloss = BCELoss(weight, size_average) 
        self.edge_loss=EDGE_LOSS()
        self.bce1 = BCELoss(weight, size_average) 
        
    def forward(self, *inputs):
        if not self.se_loss and not self.aux:
            return super(SegmentationLosses, self).forward(*inputs)
        elif not self.se_loss:
            pred1, pred2, target = tuple(inputs)
            loss1 = super(SegmentationLosses, self).forward(pred1, target)
            loss2 = super(SegmentationLosses, self).forward(pred2, target)
            return loss1 + self.aux_weight * loss2
        elif not self.aux:
            
            pred, se_pred, target = tuple(inputs)
            se_target = self._get_batch_label_vector(target, nclass=self.nclass).type_as(pred)
            loss1 = super(SegmentationLosses, self).forward(pred, target)
            loss2 = self.bceloss(torch.sigmoid(se_pred), se_target)
            return loss1 + self.se_weight * loss2
        else:
            
            #pred1,target=tuple(inputs)
            pred1, se_pred, pred2, edge, target = tuple(inputs)
            # pred1, se_pred, pred2, target = tuple(inputs)
            se_target = self._get_batch_label_vector(target, nclass=self.nclass).type_as(pred1)
            loss1 = super(SegmentationLosses, self).forward(pred1, target)
            loss2 = super(SegmentationLosses, self).forward(pred2, target)
            loss3 = self.bceloss(torch.sigmoid(se_pred), se_target)
            loss4 = self.edge_loss(edge,target)
            return loss1 + self.aux_weight * loss2 + self.se_weight * loss3 + 0.5*loss4
            # return loss1 + self.aux_weight * loss2 + self.se_weight * loss3
            #return loss1

    @staticmethod
    def _get_batch_label_vector(target, nclass):
        # target is a 3D Variable BxHxW, output is 2D BxnClass
        batch = target.size(0)
        tvect = Variable(torch.zeros(batch, nclass))
        for i in range(batch):
            hist = torch.histc(target[i].cpu().data.float(), 
                               bins=nclass, min=0,
                               max=nclass-1)
            vect = hist>0
            tvect[i] = vect
        return tvect
import random
class EDGE_LOSS(nn.Module):
    def __init__(self):
        super(EDGE_LOSS, self).__init__()
        self.x_filter = torch.from_numpy(np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])).float().unsqueeze(0).unsqueeze(0)
        self.y_filter = torch.from_numpy(np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])).float().unsqueeze(0).unsqueeze(0)
    def forward(self,outs, targets,only_loss=True):
        # x_f = self.x_filter.cuda()
        # y_f = self.y_filter.cuda()
        target = targets.unsqueeze(1).type('torch.FloatTensor').cuda()
        g2_x = F.conv2d(target, self.x_filter.cuda(), padding=1)
        g2_y = F.conv2d(target, self.y_filter.cuda(), padding=1)

        g_2 = torch.sqrt(torch.pow(g2_x, 2) + torch.pow(g2_y, 2))
        # nnn=random.randint(1,30)
        # y = g_2.clone()
        # save_image(y[0,:,:,:], '/home/mspl/Desktop/改/改fastfcn_edge_3/experiments/segmentation/123/GT{}.png'.format(nnn))
        # y1 = outs.clone()
        # return F.binary_cross_entropy(torch.sigmoid(outs),g_2)
        return torch.mean((outs - g_2).pow(2))

class Normalize(Module):
    r"""Performs :math:`L_p` normalization of inputs over specified dimension.

    Does:

    .. math::
        v = \frac{v}{\max(\lVert v \rVert_p, \epsilon)}

    for each subtensor v over dimension dim of input. Each subtensor is
    flattened into a vector, i.e. :math:`\lVert v \rVert_p` is not a matrix
    norm.

    With default arguments normalizes over the second dimension with Euclidean
    norm.

    Args:
        p (float): the exponent value in the norm formulation. Default: 2
        dim (int): the dimension to reduce. Default: 1
    """
    def __init__(self, p=2, dim=1):
        super(Normalize, self).__init__()
        self.p = p
        self.dim = dim

    def forward(self, x):
        return F.normalize(x, self.p, self.dim, eps=1e-8)


class PyramidPooling(Module):
    """
    Reference:
        Zhao, Hengshuang, et al. *"Pyramid scene parsing network."*
    """
    def __init__(self, in_channels, norm_layer, up_kwargs):
        super(PyramidPooling, self).__init__()
        self.pool1 = AdaptiveAvgPool2d(1)
        self.pool2 = AdaptiveAvgPool2d(2)
        self.pool3 = AdaptiveAvgPool2d(3)
        self.pool4 = AdaptiveAvgPool2d(6)

        out_channels = int(in_channels/4)
        self.conv1 = Sequential(Conv2d(in_channels, out_channels, 1, bias=False),
                                norm_layer(out_channels),
                                ReLU(True))
        self.conv2 = Sequential(Conv2d(in_channels, out_channels, 1, bias=False),
                                norm_layer(out_channels),
                                ReLU(True))
        self.conv3 = Sequential(Conv2d(in_channels, out_channels, 1, bias=False),
                                norm_layer(out_channels),
                                ReLU(True))
        self.conv4 = Sequential(Conv2d(in_channels, out_channels, 1, bias=False),
                                norm_layer(out_channels),
                                ReLU(True))
        # bilinear upsample options
        self._up_kwargs = up_kwargs

    def forward(self, x):
        _, _, h, w = x.size()
        feat1 = F.upsample(self.conv1(self.pool1(x)), (h, w), **self._up_kwargs)
        feat2 = F.upsample(self.conv2(self.pool2(x)), (h, w), **self._up_kwargs)
        feat3 = F.upsample(self.conv3(self.pool3(x)), (h, w), **self._up_kwargs)
        feat4 = F.upsample(self.conv4(self.pool4(x)), (h, w), **self._up_kwargs)
        return torch.cat((x, feat1, feat2, feat3, feat4), 1)


class SeparableConv2d(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, padding=1, dilation=1, bias=False, BatchNorm=nn.BatchNorm2d):
        super(SeparableConv2d, self).__init__()

        #self.expand = nn.Conv2d(inplanes, inplanes*6, kernel_size=1, bias=False)
        #self.exbn=BatchNorm(inplanes*6)
        #self.relu6=nn.ReLU6(inplace=True)

        self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size, stride, padding, dilation, groups=inplanes, bias=bias)
        self.bn = BatchNorm(inplanes)
        self.pointwise = nn.Conv2d(inplanes, planes, 1, 1, 0, 1, 1, bias=bias)

        #self.reduce = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        #self.rebn=BatchNorm(planes)
        
    def forward(self, x):

        #ori=x
        #x=self.expand(x)
        #x=self.exbn(x)
        #x=self.relu6(x)

        x = self.conv1(x)
        x = self.bn(x)
        #x=self.relu6(x)
        x = self.pointwise(x)
        #x=self.rebn(x)

        #x=x+self.rebn(self.reduce(ori))

        return x


# class JPU(nn.Module):
#     def __init__(self, in_channels, width=512, norm_layer=None, up_kwargs=None):
#         super(JPU, self).__init__()
#         self.up_kwargs = up_kwargs
#         self._se_reduce = nn.Conv2d(
#             in_channels=width*3, out_channels=width, kernel_size=1)
#         self._se_expand = nn.Conv2d(
#             in_channels=width, out_channels=width*3, kernel_size=1)
#         self.act=nn.ReLU(inplace=True)
#         self.conv5 = nn.Sequential(
#             nn.Conv2d(in_channels[-1], width, 3, padding=1, bias=False),
#             norm_layer(width),
#             nn.ReLU(inplace=True))
#         self.conv4 = nn.Sequential(
#             nn.Conv2d(in_channels[-2], width, 3, padding=1, bias=False),
#             norm_layer(width),
#             nn.ReLU(inplace=True))
#         self.conv3 = nn.Sequential(
#             nn.Conv2d(in_channels[-3], width, 3, padding=1, bias=False),
#             norm_layer(width),
#             nn.ReLU(inplace=True))
#         #self.conv2 = nn.Sequential(
#         #    nn.Conv2d(in_channels[-4], width, 3, padding=1, bias=False),
#         #    norm_layer(width),
#         #    nn.ReLU(inplace=True))
#         # self.conv5 = nn.Sequential(
#         #     nn.Conv2d(width, width, 3, padding=1, bias=False),
#         #     norm_layer(width),
#         #     nn.ReLU(inplace=True),
#         #     nn.Conv2d(width, width, 3, padding=1, bias=False))
#         # self.conv4 = nn.Sequential(
#         #     nn.Conv2d(width, width, 3, padding=1, bias=False),
#         #     norm_layer(width),
#         #     nn.ReLU(inplace=True),
#         #     nn.Conv2d(width, width, 3, padding=1, bias=False))
#         # self.conv3 = nn.Sequential(
#         #     nn.Conv2d(width, width, 3, padding=1, bias=False),
#         #     norm_layer(width),
#         #     nn.ReLU(inplace=True),
#         #     nn.Conv2d(width, width, 3, padding=1, bias=False))

#         # self.conv113=nn.Conv2d(in_channels[-3], width, 1, bias=False)
#         # self.conv114=nn.Conv2d(in_channels[-2], width, 1, bias=False)
#         # self.conv115=nn.Conv2d(in_channels[-1], width, 1, bias=False)
#         #elf.relu=nn.ReLU(inplace=True)
#         self._se_reduce_ed = nn.Conv2d(
#             in_channels=width*4, out_channels=width, kernel_size=1)
#         self._se_expand_ed = nn.Conv2d(
#             in_channels=width, out_channels=width*4, kernel_size=1)
#         self.dilation1 = nn.Sequential(SeparableConv2d(4*width, width, kernel_size=3, padding=1, dilation=1, bias=False),
#                                        norm_layer(width),
#                                        nn.ReLU(inplace=True))
#         self.dilation2 = nn.Sequential(SeparableConv2d(4*width, width, kernel_size=3, padding=2, dilation=2, bias=False),
#                                        norm_layer(width),
#                                        nn.ReLU(inplace=True))
#         self.dilation3 = nn.Sequential(SeparableConv2d(4*width, width, kernel_size=3, padding=4, dilation=4, bias=False),
#                                        norm_layer(width),
#                                        nn.ReLU(inplace=True))
#         self.dilation4 = nn.Sequential(SeparableConv2d(4*width, width, kernel_size=3, padding=8, dilation=8, bias=False),
#                                        norm_layer(width),
#                                        nn.ReLU(inplace=True))

#         # num_features=3*width
#         # d_feature0=128
#         # d_feature1=64
#         # # self.ASPP_3 = _DenseAsppBlock(input_num=num_features, num1=d_feature0, num2=d_feature1,
#         # #                               dilation_rate=3, drop_out=0.1, bn_start=False)

#         # self.ASPP_6 = _DenseAsppBlock(input_num=num_features, num1=d_feature0, num2=d_feature1,
#         #                               dilation_rate=6, drop_out=0.1, bn_start=True)

#         # self.ASPP_12 = _DenseAsppBlock(input_num=num_features + d_feature1 * 1, num1=d_feature0, num2=d_feature1,
#         #                                dilation_rate=12, drop_out=0.1, bn_start=True)

#         # self.ASPP_18 = _DenseAsppBlock(input_num=num_features + d_feature1 * 2, num1=d_feature0, num2=d_feature1,
#         #                                dilation_rate=18, drop_out=0.1, bn_start=True)

#         # self.ASPP_24 = _DenseAsppBlock(input_num=num_features + d_feature1 * 3, num1=d_feature0, num2=d_feature1,
#         #                                dilation_rate=24, drop_out=0.1, bn_start=True)


#         # self.dilation1 = nn.Sequential(nn.Conv2d(3*width, width, kernel_size=3, padding=1, dilation=1, bias=False),
#         #                                norm_layer(width),
#         #                                nn.ReLU(inplace=True))
#         # self.dilation2 = nn.Sequential(nn.Conv2d(3*width, width, kernel_size=3, padding=2, dilation=2, bias=False),
#         #                                norm_layer(width),
#         #                                nn.ReLU(inplace=True))
#         # self.dilation3 = nn.Sequential(nn.Conv2d(3*width, width, kernel_size=3, padding=4, dilation=4, bias=False),
#         #                                norm_layer(width),
#         #                                nn.ReLU(inplace=True))
#         # self.dilation4 = nn.Sequential(nn.Conv2d(3*width, width, kernel_size=3, padding=8, dilation=8, bias=False),
#         #                                norm_layer(width),
#         #                                nn.ReLU(inplace=True))
#         # self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
#         #                                      nn.Conv2d(3*width, width, 1, stride=1, bias=False),
#         #                                      norm_layer(width),
#         #                                      nn.ReLU())
#     # def forward(self, *inputs):
#     #     feats = [self.conv5(inputs[-1]), self.conv4(inputs[-2]), self.conv3(inputs[-3])]
#     #     _, _, h, w = feats[-1].size()
#     #     feats[-3] = F.upsample(feats[-3], (h//2, w//2), **self.up_kwargs)
#     #     feats[-2] = torch.cat([feats[-2],feats[-3]], dim=1)
#     #     feats[-2] = F.upsample(feats[-2], (h, w), **self.up_kwargs)

#     #     feat = torch.cat([feats[-2],feats[-1]], dim=1)
#     #     # x5 = self.global_avg_pool(feat)
#     #     # x5 = F.interpolate(x5, size=feat.size()[2:], mode='bilinear', align_corners=True)

#     #     feat = torch.cat([self.dilation1(feat), self.dilation2(feat), self.dilation3(feat), self.dilation4(feat)], dim=1)
#     #     return inputs[0], inputs[1], inputs[2], feat

#     # def forward(self, *inputs):
#     #     feats = [self.conv5(inputs[-1]), self.conv4(inputs[-2]), self.conv3(inputs[-3])]
#     #     _, _, h, w = feats[-1].size()
#     #     feats[-2] = F.upsample(feats[-2], (h, w), **self.up_kwargs)
#     #     feats[-3] = F.upsample(feats[-3], (h, w), **self.up_kwargs)
#     #     feat = torch.cat(feats, dim=1)
#     #     feat = torch.cat([self.dilation1(feat), self.dilation2(feat), self.dilation3(feat), self.dilation4(feat)], dim=1)
#     #     return inputs[0], inputs[1], inputs[2], feat

#     # def forward(self, *inputs):
#     #     f=[self.conv115(inputs[-1]),self.conv114(inputs[-2]),self.conv113(inputs[-3])]
#     #     f1=[self.conv5(f[0]),self.conv4(f[1]),self.conv3(f[2])]
#     #     feats=[self.relu(f[0]+f1[0]),self.relu(f[1]+f1[1]),self.relu(f[2]+f1[2])]

#     #     # feats = [self.conv5(inputs[-1]), self.conv4(inputs[-2]), self.conv3(inputs[-3])]
#     #     _, _, h, w = feats[-1].size()
#     #     feats[-2] = F.upsample(feats[-2], (h, w), **self.up_kwargs)
#     #     feats[-3] = F.upsample(feats[-3], (h, w), **self.up_kwargs)
#     #     feat = torch.cat(feats, dim=1)
#     #     feat = torch.cat([self.dilation1(feat), self.dilation2(feat), self.dilation3(feat), self.dilation4(feat)], dim=1)
#     #     return inputs[0], inputs[1], inputs[2], feat

#     # def forward(self, *inputs):
#     #     feats = [self.conv5(inputs[-1]), self.conv4(inputs[-2]), self.conv3(inputs[-3])]
#     #     _, _, h, w = feats[-1].size()
#     #     feats[-2] = F.upsample(feats[-2], (h, w), **self.up_kwargs)
#     #     feats[-3] = F.upsample(feats[-3], (h, w), **self.up_kwargs)
#     #     feat = torch.cat(feats, dim=1)

#     #     # aspp3 = self.ASPP_3(feat)
#     #     # feature = torch.cat((aspp3, feat), dim=1)

#     #     aspp6 = self.ASPP_6(feat)
#     #     feature = torch.cat((aspp6, feat), dim=1)

#     #     aspp12 = self.ASPP_12(feature)
#     #     feature = torch.cat((aspp12, feature), dim=1)

#     #     aspp18 = self.ASPP_18(feature)
#     #     feature = torch.cat((aspp18, feature), dim=1)

#     #     aspp24 = self.ASPP_24(feature)
#     #     feat = torch.cat((aspp24, feature), dim=1)

#     #     return inputs[0], inputs[1], inputs[2], feat
#     # def forward(self, *inputs):
#     #     feats = [self.conv5(inputs[-1]), self.conv4(inputs[-2]), self.conv3(inputs[-3])]
#     #     _, _, h, w = feats[-1].size()
#     #     feats[-2] = F.upsample(feats[-2], (h, w), **self.up_kwargs)
#     #     feats[-3] = F.upsample(feats[-3], (h, w), **self.up_kwargs)
#     #     feat = torch.cat(feats, dim=1)



#     #     feat = torch.cat([self.dilation1(feat), self.dilation2(feat), self.dilation3(feat), self.dilation4(feat)], dim=1)
#     #     return inputs[0], inputs[1], inputs[2], feat
#     def forward(self, *inputs):
#         feats = [self.conv5(inputs[-1]), self.conv4(inputs[-2]), self.conv3(inputs[-3])]
#         _, _, h, w = feats[-1].size()
#         feats[-2] = F.upsample(feats[-2], (h, w), **self.up_kwargs)
#         feats[-3] = F.upsample(feats[-3], (h, w), **self.up_kwargs)
#         #feats[-4] = F.upsample(feats[-4], (h, w), **self.up_kwargs)

#         feat1=feats[0]+feats[1]+feats[2]
#         feat = torch.cat(feats, dim=1)
#         feat = torch.cat([feat,feat1],dim=1)
#         #x_squeezed = F.adaptive_avg_pool2d(feat, 1)
#         #x_squeezed = self._se_expand(
#         #    self.act(self._se_reduce(x_squeezed)))
#         #feat = torch.sigmoid(x_squeezed) * feat
        
#         feat = torch.cat([self.dilation1(feat), self.dilation2(feat), self.dilation3(feat), self.dilation4(feat)], dim=1)
#         #x_squeezed = F.adaptive_avg_pool2d(feat, 1)
#         #x_squeezed = self._se_expand_ed(
#         #    self.act(self._se_reduce_ed(x_squeezed)))
#         #feat = torch.sigmoid(x_squeezed) * feat
#         #if len(inputs) >=5:
#         #    return inputs[0],inputs[1],inputs[2],inputs[3],inputs[4],inputs[5],feat
#         #else:
#         #    return inputs[0],inputs[1],inputs[2],feat
#         return inputs[0], inputs[1], inputs[2], feat
class Mean(Module):
    def __init__(self, dim, keep_dim=False):
        super(Mean, self).__init__()
        self.dim = dim
        self.keep_dim = keep_dim

    def forward(self, input):
        return input.mean(self.dim, self.keep_dim)

class _DenseAsppBlock(nn.Sequential):
    """ ConvNet block for building DenseASPP. """

    def __init__(self, input_num, num1, num2, dilation_rate, drop_out, bn_start=True):
        super(_DenseAsppBlock, self).__init__()
        if bn_start:
            self.add_module('norm_1', nn.BatchNorm2d(input_num, momentum=0.0003)),

        self.add_module('relu_1', nn.ReLU(inplace=True)),
        self.add_module('conv_1', nn.Conv2d(in_channels=input_num, out_channels=num1, kernel_size=1)),

        self.add_module('norm_2', nn.BatchNorm2d(num1, momentum=0.0003)),
        self.add_module('relu_2', nn.ReLU(inplace=True)),
        self.add_module('conv_2', nn.Conv2d(in_channels=num1, out_channels=num2, kernel_size=3,
                                            dilation=dilation_rate, padding=dilation_rate)),

        self.drop_rate = drop_out

    def forward(self, _input):
        feature = super(_DenseAsppBlock, self).forward(_input)

        if self.drop_rate > 0:
            feature = F.dropout2d(feature, p=self.drop_rate, training=self.training)

        return feature
