

from .utils import IntermediateLayerGetter
from ._deeplab import DeepLabHead, DeepLabHeadV3Plus, DeepLabV3
from .backbone import (
    resnet,
    mobilenetv2,
    hrnetv2,
    xception,
    efficientformerv2
)
'''
from utils import IntermediateLayerGetter
from _deeplab import DeepLabHead, DeepLabHeadV3Plus, DeepLabV3
from backbone import (
    resnet,
    mobilenetv2,
    hrnetv2,
    xception,
    efficientformerv2
)
'''
def _segm_hrnet(name, backbone_name, num_classes, pretrained_backbone):

    backbone = hrnetv2.__dict__[backbone_name](pretrained_backbone)
    # HRNetV2 config:
    # the final output channels is dependent on highest resolution channel config (c).
    # output of backbone will be the inplanes to assp:
    hrnet_channels = int(backbone_name.split('_')[-1])
    inplanes = sum([hrnet_channels * 2 ** i for i in range(4)])
    low_level_planes = 256 # all hrnet version channel output from bottleneck is the same
    aspp_dilate = [12, 24, 36] # If follow paper trend, can put [24, 48, 72].

    if name=='deeplabv3plus':
        return_layers = {'stage4': 'out', 'layer1': 'low_level'}
        classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes, aspp_dilate)
    elif name=='deeplabv3':
        return_layers = {'stage4': 'out'}
        classifier = DeepLabHead(inplanes, num_classes, aspp_dilate)

    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers, hrnet_flag=True)
    model = DeepLabV3(backbone, classifier)
    return model

def _segm_resnet(name, backbone_name, num_classes, output_stride, pretrained_backbone):

    if output_stride==8:
        replace_stride_with_dilation=[False, True, True]
        aspp_dilate = [12, 24, 36]
    else:
        replace_stride_with_dilation=[False, False, True]
        aspp_dilate = [6, 12, 18]

    backbone = resnet.__dict__[backbone_name](
        pretrained=pretrained_backbone,
        replace_stride_with_dilation=replace_stride_with_dilation)
    
    inplanes = 2048
    low_level_planes = 256

    if name=='deeplabv3plus':
        return_layers = {'layer4': 'out', 'layer1': 'low_level'}
        classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes, aspp_dilate)
    elif name=='deeplabv3':
        return_layers = {'layer4': 'out'}
        classifier = DeepLabHead(inplanes , num_classes, aspp_dilate)
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    model = DeepLabV3(backbone, classifier)
    return model


def _segm_xception(name, backbone_name, num_classes, output_stride, pretrained_backbone):
    if output_stride==8:
        replace_stride_with_dilation=[False, False, True, True]
        aspp_dilate = [12, 24, 36]
    else:
        replace_stride_with_dilation=[False, False, False, True]
        aspp_dilate = [6, 12, 18]
    
    backbone = xception.xception(pretrained= 'imagenet' if pretrained_backbone else False, replace_stride_with_dilation=replace_stride_with_dilation)
    
    inplanes = 2048
    low_level_planes = 128
    
    if name=='deeplabv3plus':
        return_layers = {'conv4': 'out', 'block1': 'low_level'}
        classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes, aspp_dilate)
    elif name=='deeplabv3':
        return_layers = {'conv4': 'out'}
        classifier = DeepLabHead(inplanes , num_classes, aspp_dilate)
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
    model = DeepLabV3(backbone, classifier)
    return model


def _segm_mobilenet(name, backbone_name, num_classes, output_stride, pretrained_backbone):
    if output_stride==8:
        aspp_dilate = [12, 24, 36]
    else:
        aspp_dilate = [6, 12, 18]

    backbone = mobilenetv2.mobilenet_v2(pretrained=pretrained_backbone, output_stride=output_stride)
    
    # rename layers
    backbone.low_level_features = backbone.features[0:4]
    backbone.high_level_features = backbone.features[4:-1]
    backbone.features = None
    backbone.classifier = None

    inplanes = 320
    low_level_planes = 24
    
    if name=='deeplabv3plus':
        return_layers = {'high_level_features': 'out', 'low_level_features': 'low_level'}
        classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes, aspp_dilate)
    elif name=='deeplabv3':
        return_layers = {'high_level_features': 'out'}
        classifier = DeepLabHead(inplanes , num_classes, aspp_dilate)
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    model = DeepLabV3(backbone, classifier)
    return model

def _segm_effiformer(name, backbone_name, num_classes, output_stride, pretrained_backbone):
    if output_stride==8:
        aspp_dilate = [12,24,36]
    else:
        aspp_dilate = [6,12,18]
        replace_stride_with_dilation=[False, False, True]
    '''
    backbone = efficientformerv2.efficientformerv2_l(pretrained=pretrained_backbone,outputstride = output_stride)
    '''
    backbone = efficientformerv2.efficientformerv2_l(pretrained="pretrained/eformer_l_450.pth",outputstride = output_stride)
    
    inplanes = 384
    low_level_planes = 80
    if name =='deeplabv3plus':
        return_layers = {'network': 'out', 'patch_embed': 'low_level'}
        classifier = DeepLabHead(inplanes , num_classes, aspp_dilate)
        print("NaneError")
    model = DeepLabV3(backbone,classifier)
    return model

def _segm_effiformers2(name,backbone_name,num_classes,output_stride,pretrained_backbone):
    if output_stride==8:
        aspp_dilate = [12,24,36]
    else:
        aspp_dilate = [6,12,18]
        replace_stride_with_dilation=[False,False,True]
    backbone = efficientformerv2.efficientformerv2_s2("pretrained/eformer_s2_450.pth",outputstride=output_stride)
    
    inplanes = 288
    low_level_planes = 64
    if name == 'deeplabv3plus':
        return_layers ={'network': 'out', 'patch_embed': 'low_level'}
        classifier = DeepLabHeadV3Plus(inplanes,low_level_planes,num_classes,aspp_dilate)
    elif name == 'deeplabv3':
        return_layers = {"network":"out"}
        classifier = DeepLabHead(inplanes,num_classes,aspp_dilate)
        
    model = DeepLabV3(backbone,classifier)
    return model

def _load_model(arch_type, backbone, num_classes, output_stride, pretrained_backbone):

    if backbone=='mobilenetv2':
        model = _segm_mobilenet(arch_type, backbone, num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)
    elif backbone.startswith('resnet'):
        model = _segm_resnet(arch_type, backbone, num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)
    elif backbone.startswith('hrnetv2'):
        model = _segm_hrnet(arch_type, backbone, num_classes, pretrained_backbone=pretrained_backbone)
    elif backbone=='xception':
        model = _segm_xception(arch_type, backbone, num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)
    
    elif backbone=='effiformer':
        model = _segm_effiformer(arch_type,backbone,num_classes,output_stride=output_stride,
pretrained_backbone=pretrained_backbone)
    
    elif backbone=='effiformer_s2':
        model = _segm_effiformers2(arch_type,backbone,num_classes,output_stride=output_stride,pretrained_backbone=pretrained_backbone)
    else:
        raise NotImplementedError
        
    return model


# Deeplab v3
def deeplabv3_hrnetv2_48(num_classes=21, output_stride=4, pretrained_backbone=False): # no pretrained backbone yet
    return _load_model('deeplabv3', 'hrnetv2_48', output_stride, num_classes, pretrained_backbone=pretrained_backbone)

def deeplabv3_hrnetv2_32(num_classes=21, output_stride=4, pretrained_backbone=True):
    return _load_model('deeplabv3', 'hrnetv2_32', output_stride, num_classes, pretrained_backbone=pretrained_backbone)

def deeplabv3_resnet50(num_classes=21, output_stride=8, pretrained_backbone=True):
    """Constructs a DeepLabV3 model with a ResNet-50 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('deeplabv3', 'resnet50', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)

def deeplabv3_resnet101(num_classes=21, output_stride=8, pretrained_backbone=True):
    """Constructs a DeepLabV3 model with a ResNet-101 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('deeplabv3', 'resnet101', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)

def deeplabv3_mobilenet(num_classes=21, output_stride=8, pretrained_backbone=True, **kwargs):
    """Constructs a DeepLabV3 model with a MobileNetv2 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('deeplabv3', 'mobilenetv2', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)

def deeplabv3_xception(num_classes=21, output_stride=8, pretrained_backbone=True, **kwargs):
    """Constructs a DeepLabV3 model with a Xception backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('deeplabv3', 'xception', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)


# Deeplab v3+
def deeplabv3plus_hrnetv2_48(num_classes=21, output_stride=4, pretrained_backbone=False): # no pretrained backbone yet
    return _load_model('deeplabv3plus', 'hrnetv2_48', num_classes, output_stride, pretrained_backbone=pretrained_backbone)

def deeplabv3plus_hrnetv2_32(num_classes=21, output_stride=4, pretrained_backbone=True):
    return _load_model('deeplabv3plus', 'hrnetv2_32', num_classes, output_stride, pretrained_backbone=pretrained_backbone)

def deeplabv3plus_resnet50(num_classes=21, output_stride=8, pretrained_backbone=True):
    """Constructs a DeepLabV3 model with a ResNet-50 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('deeplabv3plus', 'resnet50', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)


def deeplabv3plus_resnet101(num_classes=21, output_stride=8, pretrained_backbone=True):
    """Constructs a DeepLabV3+ model with a ResNet-101 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('deeplabv3plus', 'resnet101', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)


def deeplabv3plus_mobilenet(num_classes=21, output_stride=8, pretrained_backbone=True):
    """Constructs a DeepLabV3+ model with a MobileNetv2 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('deeplabv3plus', 'mobilenetv2', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)

def deeplabv3plus_xception(num_classes=21, output_stride=8, pretrained_backbone=True):
    """Constructs a DeepLabV3+ model with a Xception backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('deeplabv3plus', 'xception', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)

def deeplabv3plus_effiformer(num_classes=8,output_stride = 8,pretrained_backbone = True):
    #print(pretrained_backbone)
    return _load_model('deeplabv3plus','effiformer',num_classes,output_stride=output_stride,
pretrained_backbone=pretrained_backbone)

def deeplabv3plus_effiformer_s2(num_classes=8,output_stride=8,pretrained_backbone=True):
    return _load_model('deeplabv3','effiformer_s2',num_classes,output_stride=output_stride,pretrained_backbone=pretrained_backbone)
if __name__ == "__main__":
    
    #model = deeplabv3plus_effiformer()
    model = deeplabv3_mobilenet()
    model = model.eval()
    import numpy as np
    import torch
    import time
    x = np.zeros((1,3,224,224)).astype(np.float32)
    #print(x.shape)
    #print(model)
    x = torch.tensor(x)
    t1 = time.time()    
    result = model(x)
    t2 = time.time()
    #torch.save("13123.pt",model)
    print(t2-t1,"sec")
    print(result.shape)
    