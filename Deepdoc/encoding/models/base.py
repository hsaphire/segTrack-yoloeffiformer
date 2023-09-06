###########################################################################
# Created by: Hang Zhang 
# Email: zhang.hang@rutgers.edu 
# Copyright (c) 2017 @dfpm
###########################################################################

import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.parallel.scatter_gather import scatter
from torch.nn.parallel.data_parallel import DataParallel

# from ..nn import JPU
# from .. import dilated as resnet
from Deepdoc.encoding.efficientnet import EfficientNet
from Deepdoc.encoding.utils import batch_pix_accuracy, batch_intersection_union
from Deepdoc.encoding.mobilenet import MobileNetV2


up_kwargs = {'mode': 'bilinear', 'align_corners': True}

__all__ = ['BaseNet', 'MultiEvalModule']

class MobileNetConv2d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MobileNetConv2d, self).__init__()
        
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x
    
class DJPM(nn.Module):
    def __init__(self, in_channels, width=256, norm_layer=nn.BatchNorm2d, up_kwargs=None):
        super(DJPM, self).__init__()
        self.up_kwargs = up_kwargs
        """
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels[-1], width, 3, padding=1, bias=False),
            norm_layer(width),
            nn.ReLU(inplace=True))#第6個input
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels[-2], width, 3, padding=1, bias=False),
            norm_layer(width),
            nn.ReLU(inplace=True))#第5個input
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels[-3], width, 3, padding=1, bias=False),
            norm_layer(width),
            nn.ReLU(inplace=True))#第4個input
        """
        self.conv5 = nn.Sequential(
            MobileNetConv2d(in_channels[-1], width),
            norm_layer(width),
            nn.ReLU(inplace=True))#第6個input
        self.conv4 = nn.Sequential(
            MobileNetConv2d(in_channels[-2], width),
            norm_layer(width),
            nn.ReLU(inplace=True))#第5個input
        self.conv3 = nn.Sequential(
            MobileNetConv2d(in_channels[-3], width),
            norm_layer(width),
            nn.ReLU(inplace=True))#第4個input
        
        ratio_width=8*width
        self.dilation1 = nn.Sequential(# pw
                                        nn.Conv2d(4*width, ratio_width, 1, 1, 0, bias=False),
                                        nn.BatchNorm2d(ratio_width),
                                        nn.ReLU6(inplace=True),
                                        # dw
                                        nn.Conv2d(ratio_width, ratio_width, 3, 1, padding=1, dilation=1,groups= ratio_width, bias=False),
                                        nn.BatchNorm2d(ratio_width),
                                        nn.ReLU6(inplace=True),
                                        # pw-linear
                                        nn.Conv2d(ratio_width, width, 1, 1, 0, bias=False),
                                        nn.BatchNorm2d(width))
        self.dilation2 = nn.Sequential(# pw
                                        nn.Conv2d(4*width,ratio_width, 1, 1, 0, bias=False),
                                        nn.BatchNorm2d(ratio_width),
                                        nn.ReLU6(inplace=True),
                                        # dw
                                        nn.Conv2d(ratio_width,ratio_width, 3, 1, padding=2, dilation=2,groups= ratio_width, bias=False),
                                        nn.BatchNorm2d( ratio_width),
                                        nn.ReLU6(inplace=True),
                                        # pw-linear
                                        nn.Conv2d( ratio_width, width, 1, 1, 0, bias=False),
                                        nn.BatchNorm2d(width))
        self.dilation3 = nn.Sequential(# pw
                                        nn.Conv2d(4*width, ratio_width, 1, 1, 0, bias=False),
                                        nn.BatchNorm2d(ratio_width),
                                        nn.ReLU6(inplace=True),
                                        # dw
                                        nn.Conv2d(ratio_width, ratio_width, 3, 1, padding=4, dilation=4,groups= ratio_width, bias=False),
                                        nn.BatchNorm2d( ratio_width),
                                        nn.ReLU6(inplace=True),
                                        # pw-linear
                                        nn.Conv2d( ratio_width, width, 1, 1, 0, bias=False),
                                        nn.BatchNorm2d(width))
        self.dilation4 = nn.Sequential(# pw
                                        nn.Conv2d(4*width,ratio_width, 1, 1, 0, bias=False),
                                        nn.BatchNorm2d(ratio_width),
                                        nn.ReLU6(inplace=True),
                                        # dw
                                        nn.Conv2d(ratio_width,ratio_width, 3, 1, padding=8, dilation=8,groups= ratio_width, bias=False),
                                        nn.BatchNorm2d(ratio_width),
                                        nn.ReLU6(inplace=True),
                                        # pw-linear
                                        nn.Conv2d(ratio_width, width, 1, 1, 0, bias=False),
                                        nn.BatchNorm2d(width))
        self.conv6 = nn.Sequential(
            nn.Conv2d(4*width, width, 1, padding=0, bias=False),
            norm_layer(width),
            nn.ReLU(inplace=True))
        # self.pooling1 = nn.MaxPool2d(2, stride=2)
        """
        self.conv8 = nn.Sequential(
            nn.Conv2d(5*width+in_channels[-4], in_channels[-4], 3, padding=1, bias=False),
            norm_layer(in_channels[-4]),
            nn.ReLU(inplace=True))
        self.conv9 = nn.Sequential(
            nn.Conv2d(5*width+in_channels[-5], in_channels[-5], 3, padding=1, bias=False),
            norm_layer(in_channels[-5]),
            nn.ReLU(inplace=True))
        # self.conv9 = nn.Sequential(
        #     nn.Conv2d(5*width+in_channels[-6], in_channels[-6], 3, padding=1, bias=False),
        #     norm_layer(in_channels[-6]),
        #     nn.ReLU(inplace=True))
        """
        self.conv8 = nn.Sequential(
            MobileNetConv2d(5*width+in_channels[-4], in_channels[-4]),
            norm_layer(in_channels[-4]),
            nn.ReLU(inplace=True))
        self.conv9 = nn.Sequential(
            MobileNetConv2d(5*width+in_channels[-5], in_channels[-5]),
            norm_layer(in_channels[-5]),
            nn.ReLU(inplace=True))
        
    def forward(self, input_1, input_2, input_3, input_4, input_5):
        # _, _, h_0, w_0 = input_0.size()
        _, _, h_1, w_1 = input_1.size()
        _, _, h_2, w_2 = input_2.size()

        feats = [self.conv5(input_5), self.conv4(input_4), self.conv3(input_3)]
        _, _, h, w = feats[-1].size()
        feats[-2] = F.upsample(feats[-2], (h, w), **self.up_kwargs)
        feats[-3] = F.upsample(feats[-3], (h, w), **self.up_kwargs)

        feat1 = feats[0]+feats[1]+feats[2]
        feat = torch.cat(feats, dim=1)
        feat = torch.cat([feat,feat1],dim=1)

        tmp=[self.dilation1(feat), self.dilation2(feat), self.dilation3(feat), self.dilation4(feat)]
        feat = torch.cat([tmp[0],tmp[1],tmp[2],tmp[3]], dim=1)#4*width
        feat2 = self.conv6(feat)#1*width
        feat = torch.cat([feat,feat2],dim=1)#5*width d的結尾
        # pool1 = feat#d部份結尾

        # pool1 = F.interpolate(feat, size=[h_0, w_0], mode="bilinear")
        pool1 = F.interpolate(feat, size=[h_1, w_1], mode="bilinear")
        pool2 = F.interpolate(feat, size=[h_2, w_2], mode="bilinear")
        # print(pool1.size())
        # print(input_1.size())
        # print(pool2.size())
        # print(input_2.size())
        # pool0 = torch.cat([pool1, input_0],dim=1)
        pool1 = torch.cat([pool1, input_1],dim=1)
        pool2 = torch.cat([pool2, input_2],dim=1)
        # pool3 = torch.cat([pool3, input_3],dim=1)

        # pool0 = self.conv9(pool0)
        pool2 = self.conv8(pool2)
        pool1 = self.conv9(pool1)
        
        return pool1, pool2


class BaseNet(nn.Module):
    #jpu=True
    def __init__(self, nclass, backbone, aux, se_loss, jpu=True, dilated=False, norm_layer=None,
                 base_size=520, crop_size=480, mean=[.485, .456, .406],
                 std=[.229, .224, .225], root='/encoding/architecture', **kwargs):
        super(BaseNet, self).__init__()
        self.nclass = nclass
        self.aux = aux
        self.se_loss = se_loss
        self.mean = mean
        self.std = std
        self.base_size = base_size
        self.crop_size = crop_size

        # copying modules from pretrained models
        if backbone == 'resnet50':
            self.pretrained = resnet.resnet50(pretrained=True, dilated=dilated,
                                              norm_layer=norm_layer, root=root)
        elif backbone == 'resnet101':
            self.pretrained = resnet.resnet101(pretrained=True, dilated=dilated,
                                               norm_layer=norm_layer, root=root)
        elif backbone == 'resnet152':
            self.pretrained = resnet.resnet152(pretrained=True, dilated=dilated,
                                               norm_layer=norm_layer, root=root)
        elif backbone == 'densenet':
            self.pretrained = resnet.densenet121(pretrained=True)
        elif backbone == 'efficientnet':
            self.pretrained = EfficientNet.from_pretrained(model_name='efficientnet-b3')
        elif backbone == 'mobilenet':
            Net = MobileNetV2()
            Net.load_state_dict(torch.load('./encoding/architecture/mobilenetv2-c5e733a8.pth')) 
            self.pretrained = Net
            print("yo")
        else:
            raise RuntimeError('unknown backbone: {}'.format(backbone))
        # bilinear upsample options
        #print(self.pretrained)
        # if jpu:
        #     if backbone == 'densenet':
        #          self.jpu = JPU([512, 1024, 1024], width=256, norm_layer=norm_layer, up_kwargs=up_kwargs) if jpu else None
        #          #self.jpu = JPU([512, 1280, 1664], width=256,
        #          #               norm_layer=norm_layer, up_kwargs=up_kwargs) if jpu else None
        #     elif backbone == 'efficientnet':
        #         self.jpu =  JPU([48,32,40], width=10,   # -4 -3 -1 layer and -1 layer / 4 
        #                        norm_layer=norm_layer, up_kwargs=up_kwargs) if jpu else None
        #     else:
        #         self.jpu = JPU([512, 1024, 2048], width=512, norm_layer=norm_layer, up_kwargs=up_kwargs) if jpu else None
        
            #print('aaaaa')
        self.jpu=False
        #DJPM
        # self.djpm = DJPM([24, 48, 120, 208, 352], width=256, up_kwargs=up_kwargs)#最左邊3個與最右邊3個，按照順序放（左到右）b2
        self.djpm = DJPM([32, 48, 136, 232, 384], width=256, up_kwargs=up_kwargs)#最左邊3個與最右邊3個，按照順序放（左到右) b3
        # self.djpm = DJPM([24, 40, 112, 192, 320], width=256, up_kwargs=up_kwargs)#最左邊3個與最右邊3個，按照順序放（左到右) b1
        # self.djpm = DJPM([32, 56, 160, 272, 448], width=256, up_kwargs=up_kwargs)#最左邊3個與最右邊3個，按照順序放（左到右) b4
        # self.djpm = DJPM([40, 64, 176, 304, 512], width=256, up_kwargs=up_kwargs)#最左邊3個與最右邊3個，按照順序放（左到右) b5
        # self.djpm = DJPM([32, 136, 232, 384], width=256, up_kwargs=up_kwargs)#最左邊3個與最右邊3個，按照順序放（左到右) b3
        self._up_kwargs = up_kwargs
        self.backbone = backbone

    def base_forward(self, x): ######  !backbone
        if self.backbone=='densenet':
            x = self.pretrained.features.conv0(x)
            x = self.pretrained.features.norm0(x)
            x = self.pretrained.features.relu0(x)
            x = self.pretrained.features.pool0(x)
            #print(x.shape)
            c1 = self.pretrained.features.denseblock1(x)#c1 8 256 120 120
            #print(c1.shape,'1')
            c2 = self.pretrained.features.transition1(c1)
            c2 = self.pretrained.features.denseblock2(c2)#c2 8 512 60 60
            #print(c2.shape,'1')
            c3 = self.pretrained.features.transition2(c2)
            c3 = self.pretrained.features.denseblock3(c3)#c3 8 1024 30 30
            #print(c3.shape,'1')
            c4 = self.pretrained.features.transition3(c3)
            c4 = self.pretrained.features.denseblock4(c4)#c4 8 1024 15 15
            #print(c4.shape,'1')
        elif self.backbone == 'efficientnet':
           
            ori=x
            
            x = self.pretrained._swish(
                self.pretrained._bn0(self.pretrained._conv_stem(x)))
            
            layer=[]
            first=x
            i=0
            for idx, block in enumerate(self.pretrained._blocks):
                drop_connect_rate = self.pretrained._global_params.drop_connect_rate
                if drop_connect_rate:
                    drop_connect_rate *= float(idx) / len(self.pretrained._blocks)
                
                x = block(x, drop_connect_rate=drop_connect_rate)
                #print('==============='+ str(i) +'================')
                #print(x.size())
                if idx+1 == self.pretrained._layer[i]:
                    #if i>=1 and i<=5:
                    #    x=self.pretrained._senet[i-1](x)
                    layer.append(x)
                    #print('===============layer================')
                    #print(self.pretrained._layer[i])
                    #print(x.size())
                    if i==6:
                        # print(len(layer))
                        out_1, out_2 = self.djpm(layer[1], layer[2], layer[4], layer[5], layer[6])
                        # out_1 = self.djpm(layer[1], layer[4], layer[5], layer[6])

                    if i==8 or i==9 or i==11 or i==12:
                        imasize=layer[12-i].size()[2:]
                        x = F.upsample(x, imasize, **up_kwargs)
                        #print('ccccccccccccccccccccccc')
                        #print(x.size())
                    # if i==7:
                    #     x = torch.cat([x, layer[12-i]], dim=1)
                    #     # x = torch.cat([x, out_2], dim=1)
                    # elif i==8:
                    #     # print(x.size())
                    #     # print(out_1.size())
                    #     x = torch.cat([x, out_1], dim=1)
                    # elif i==9:
                    #     x = torch.cat([x, out_2], dim=1)
                    # elif i==10:
                    #     x = torch.cat([x, out_3], dim=1)
                    # elif i>=11 and i <12:
                    #     x = torch.cat([x, layer[12-i]], dim=1)

                    if i >= 7 and i < 10:
                        #x = torch.cat([x, self.pretrained._senet[11-i](layer[12-i])], dim=1)
                        #se = self.pretrained._senet[11-i](layer[12-i])
                        #x = torch.cat([x, self.pretrained._att[11-i](x,layer[12-i])], dim=1)
                        x = torch.cat([x, layer[12-i]], dim=1)
                    # elif i==9:
                    #     x = torch.cat([x, out_3], dim=1)

                    elif i==10:
                        # print(x.size())
                        # print(out_1.size())
                        x = torch.cat([x, out_2], dim=1)
                    elif i==11:
                        x = torch.cat([x, out_1], dim=1)
                        # x = torch.cat([x, layer[12-i]], dim=1)
                    
                    i+=1
            #x=torch.cat([x,first],dim=1)
            #x = self.pretrained._swish(
            #    self.pretrained._bn0_de(self.pretrained._conv_stem_de(x)))
            #imasize = (ori.size()[2:])
            #x=F.upsample(x,imasize,**up_kwargs)
            #layer.append(x)
        elif self.backbone == 'mobilenet':
            x = self.prtrained.features(x)
            x = self.prtrained.conv(x)
            x = self.prtrained.avgpool(x)
            x = x.prtrained.view(x.size(0), -1)
            x = self.prtrained.classifier(x)
            
            
        else:
            x = self.pretrained.conv1(x)
            x = self.pretrained.bn1(x)
            x = self.pretrained.relu(x)
            x = self.pretrained.maxpool(x)
            c1 = self.pretrained.layer1(x)#c1 8 256 120 120
            c2 = self.pretrained.layer2(c1)#c2 8 512 60 60
            c3 = self.pretrained.layer3(c2)#c3 8 1024 30 30
            c4 = self.pretrained.layer4(c3)#c4 8 2048 15 15

        # if self.jpu:
        #     if self.backbone == 'efficientnet':
        #         return self.jpu(layer[-6],layer[-4],layer[-3],layer[-1])
        #     else:
        #         return self.jpu(c1, c2, c3, c4)
        # else:
        if self.backbone == 'efficientnet':
            
            return layer[-3], layer[-5], layer[-3], layer[-1]
            
        else:
            return c1, c2, c3, c4

    def evaluate(self, x, target=None):
        pred = self.forward(x)
        if isinstance(pred, (tuple, list)):
            pred = pred[0]
        if target is None:
            return pred
        correct, labeled = batch_pix_accuracy(pred.data, target.data)
        inter, union = batch_intersection_union(pred.data, target.data, self.nclass)
        return correct, labeled, inter, union


class MultiEvalModule(DataParallel):
    """Multi-size Segmentation Eavluator"""
    def __init__(self, module, nclass, device_ids=None, flip=True,
                 scales=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75]):
        super(MultiEvalModule, self).__init__(module, device_ids)
        self.nclass = nclass
        self.base_size = module.base_size
        self.crop_size = module.crop_size
        self.scales = scales
        self.flip = flip
        print('MultiEvalModule: base_size {}, crop_size {}'. \
            format(self.base_size, self.crop_size))

    def parallel_forward(self, inputs, **kwargs):
        """Multi-GPU Mult-size Evaluation

        Args:
            inputs: list of Tensors
        """
        #################################
        self.device_ids = []
        for i in range(3):
            for j in range (2):
                self.device_ids.append(j)
        #self.device_ids = [0, 1, 0, 1,0, 1, 0, 1, 0, 1]
        inputs = [(input.unsqueeze(0).cuda(device),)
                  for input, device in zip(inputs, self.device_ids)]
        ###############################
        #print("device_id",self.device_ids)
        #print("arfararar:",len(inputs))
        replicas = self.replicate(self, self.device_ids[:len(inputs)])
        kwargs = []
        if len(inputs) < len(kwargs):
            inputs.extend([() for _ in range(len(kwargs) - len(inputs))])
        elif len(kwargs) < len(inputs):
            kwargs.extend([{} for _ in range(len(inputs) - len(kwargs))])
        outputs = self.parallel_apply(replicas, inputs, kwargs)
        #for out in outputs:
        #    print('out.size()', out.size())
        return outputs

    def forward(self, image):
        """Mult-size Evaluation"""
        # only single image is supported for evaluation
        batch, _, h, w = image.size()
        #print("batch:",batch)
        assert(batch == 1)
        stride_rate = 2.0/3.0
        crop_size = self.crop_size
        stride = int(crop_size * stride_rate)
        with torch.cuda.device_of(image):
            scores = image.new().resize_(batch,self.nclass,h,w).zero_().cuda()

        for scale in self.scales:
            long_size = int(math.ceil(self.base_size * scale))
            if h > w:
                height = long_size
                width = int(1.0 * w * long_size / h + 0.5)
                short_size = width
            else:
                width = long_size
                height = int(1.0 * h * long_size / w + 0.5)
                short_size = height
            # resize image to current size
            cur_img = resize_image(image, height, width, **self.module._up_kwargs)
            if long_size <= crop_size:
                pad_img = pad_image(cur_img, self.module.mean,
                                    self.module.std, crop_size)
                outputs = module_inference(self.module, pad_img, self.flip)
                outputs = crop_image(outputs, 0, height, 0, width)
            else:
                if short_size < crop_size:
                    # pad if needed
                    pad_img = pad_image(cur_img, self.module.mean,
                                        self.module.std, crop_size)
                else:
                    pad_img = cur_img
                _,_,ph,pw = pad_img.size()
                assert(ph >= height and pw >= width)
                # grid forward and normalize
                h_grids = int(math.ceil(1.0 * (ph-crop_size)/stride)) + 1
                w_grids = int(math.ceil(1.0 * (pw-crop_size)/stride)) + 1
                with torch.cuda.device_of(image):
                    outputs = image.new().resize_(batch,self.nclass,ph,pw).zero_().cuda()
                    count_norm = image.new().resize_(batch,1,ph,pw).zero_().cuda()
                # grid evaluation
                for idh in range(h_grids):
                    for idw in range(w_grids):
                        h0 = idh * stride
                        w0 = idw * stride
                        h1 = min(h0 + crop_size, ph)
                        w1 = min(w0 + crop_size, pw)
                        crop_img = crop_image(pad_img, h0, h1, w0, w1)
                        # pad if needed
                        pad_crop_img = pad_image(crop_img, self.module.mean,
                                                 self.module.std, crop_size)
                        output = module_inference(self.module, pad_crop_img, self.flip)
                        outputs[:,:,h0:h1,w0:w1] += crop_image(output,
                            0, h1-h0, 0, w1-w0)
                        count_norm[:,:,h0:h1,w0:w1] += 1
                assert((count_norm==0).sum()==0)
                outputs = outputs / count_norm
                outputs = outputs[:,:,:height,:width]

            score = resize_image(outputs, h, w, **self.module._up_kwargs)
            scores += score

        return scores


def module_inference(module, image, flip=True):
    output = module.evaluate(image)
    if flip:
        fimg = flip_image(image)
        foutput = module.evaluate(fimg)
        output += flip_image(foutput)
    return output.exp()

def resize_image(img, h, w, **up_kwargs):
    return F.upsample(img, (h, w), **up_kwargs)

def pad_image(img, mean, std, crop_size):
    b,c,h,w = img.size()
    assert(c==3)
    padh = crop_size - h if h < crop_size else 0
    padw = crop_size - w if w < crop_size else 0
    pad_values = -np.array(mean) / np.array(std)
    img_pad = img.new().resize_(b,c,h+padh,w+padw)
    for i in range(c):
        # note that pytorch pad params is in reversed orders
        img_pad[:,i,:,:] = F.pad(img[:,i,:,:], (0, padw, 0, padh), value=pad_values[i])
    assert(img_pad.size(2)>=crop_size and img_pad.size(3)>=crop_size)
    return img_pad

def crop_image(img, h0, h1, w0, w1):
    return img[:,:,h0:h1,w0:w1]

def flip_image(img):
    assert(img.dim()==4)
    with torch.cuda.device_of(img):
        idx = torch.arange(img.size(3)-1, -1, -1).type_as(img).long()
    return img.index_select(3, idx)
