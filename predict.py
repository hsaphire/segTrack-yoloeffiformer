import torch.utils.data import dataset
from tqdm import tqdm
import DeepLabV3Plus-Pytorch.network
import DeepLabV3Plus-Pytorch.utils
import os
import random
import argparse
import numpy as np

from torch.utils import data
from DeepLabV3Plus-Pytorch.datasets import data
from torchvision import transforms as T
from metrics import StreamSegMetrics

import torch
import torch.nn as nn

from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
from glob import glob
######## yolo deeplab cut line ###########
import time
from pathlib import Path 
from yolov7.tracker.byte_tracker import BYTETracker
import cv2
import torch.vackends.cudnn as cudnn
from yolov7.models.experimental import attempt_load
from yolov7.utils.datasets import LoadStreams, LoadImages
from yolov7.utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from yolov7.utils.plots import plot_one_box
from yolov7.utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
from yolov7.utils.visualize import plot_tracking,plot_tracking_route
from yolov7.utils.timer import Timer


class Deeplab():
    def __init__(pretrained,opt):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.opt = opt
        self.pretrained = pretrained
        self.model = network.modeling.__dict__[self.opt.model]
        if opts.ckpt is not None and os.path.isfile(self.opts.ckpt):
            self.checkpoint = torch.load(self.opts.ckpt, map_location=torch.device('cpu'))
            self.model.load_state_dict(checkpoint["model_state"])
            self.model = nn.DataParallel(model)
            self.model.to(device)
            print("Resume model from %s" % self.opts.ckpt)
            del checkpoint
        else:
            print("[!] Retrain")
            self.model = nn.DataParallel(model)
            self.model.to(device)
        with torch.no_grad():
            self.model = model.eval()
        
    def detect(img_path):
        if self.opts.crop_val:
            transform = T.Compose([
                    T.Resize(opts.crop_size),
                    T.CenterCrop(opts.crop_size),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]),
                ])
        else:
            transform = T.Compose([
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]),
                ])
        
        ext = os.path.basename(img_path).split('.')[-1]
            img_name = os.path.basename(img_path)[:-len(ext)-1]
            img = Image.open(img_path).convert('RGB')
            img = transform(img).unsqueeze(0) # To tensor of NCHW
            img = img.to(device)
            
            pred = model(img).max(1)[1].cpu().numpy()[0] # HW
            colorized_preds = decode_fn(pred).astype('uint8')
            colorized_preds = Image.fromarray(colorized_preds)
        
        return colorized_preds
            
class Yolov7_tracker():
    
    def __init__(opt):
        self.opts = opt
        self.device = select_device(self.opt.device)  #device for yolo
        self.yolo_model = attempt_load(weights,map_location=device_yolo)
        self.stride = int(model.stride.max()) #model stride
        self.imgsz = check_img_size(imgsz,s=stride)
        
        if trace:
            self.model = TracedModel(model, device, opt.img_size)

        if half:
            self.model.half()  # to FP16
        self.names = model.module.names if hasattr(model, 'module') else model.names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in names] 
        
    def predict(img_path):
        
        
def main(opt):
    # setup dataloader
    source,weights,view_img,save_txt,imgsz,trace = opt.source,opt.weights,opt.view_img,opt.save_txt,opt.img_size,not opt.no_trace
    ave_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    yolov7_track = Yolov7_tracker(opt)
    dataset = LoadImages(source,img_size=yolov7_tracker.self.imgsz,stride=yolov7_tracker.selfstride)
    
if __name__ == '__main__':
    
    # yolo options
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    parser.add_argument("--track_thresh", type=float, default=0.25, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=120, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.95, help="matching threshold for tracking")
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")
    parser.add_argument(
        "--aspect_ratio_thresh", type=float, default=1.6,
        help="threshold for filtering out boxes of which aspect ratio are above the given value."
    )
    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
     # Deeplab Options
    parser.add_argument("--dataset", type=str, default='voc',
                        choices=['voc', 'cityscapes'], help='Name of training set')
    
    available_models = sorted(name for name in network.modeling.__dict__ if name.islower() and \
                              not (name.startswith("__") or name.startswith('_')) and callable(
                              network.modeling.__dict__[name])
                              )

    parser.add_argument("--model", type=str, default='deeplabv3plus_mobilenet',
                        choices=available_models, help='model name')
    parser.add_argument("--separable_conv", action='store_true', default=False,
                        help="apply separable conv to decoder and aspp")
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])

    # Train Options
    parser.add_argument("--save_val_results_to", default=None,
                        help="save segmentation results to the specified dir")

    parser.add_argument("--crop_val", action='store_true', default=False,
                        help='crop validation (default: False)')
    parser.add_argument("--val_batch_size", type=int, default=4,
                        help='batch size for validation (default: 4)')
    parser.add_argument("--crop_size", type=int, default=513)

    
    parser.add_argument("--ckpt", default=None, type=str,
                        help="resume from checkpoint")
    ########### option endline ######################
    
    opts = opt.parse_args()