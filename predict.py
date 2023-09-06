from torch.utils.data import dataset
from tqdm import tqdm
import DeepLabV3Plus_Pytorch.network as network
import DeepLabV3Plus_Pytorch.utils 
import os
import random
import argparse
import numpy as np
#####
from torch.utils import data
from DeepLabV3Plus_Pytorch.datasets import data
from torchvision import transforms as T
from DeepLabV3Plus_Pytorch.metrics import StreamSegMetrics
from DeepLabV3Plus_Pytorch.datasets import VOCSegmentation, Cityscapes, cityscapes
import torch
import torch.nn as nn

from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
from glob import glob
######## yolo deeplab cut line ###########
import time
from pathlib import Path 
from yolov7.tracker.mod_byte_tracker import BYTETracker
import cv2
import torch.backends.cudnn as cudnn
from yolov7.models.experimental import attempt_load
from yolov7.utils.datasets import LoadStreams, LoadImages
from yolov7.utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from yolov7.utils.plots import plot_one_box
from yolov7.utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
from yolov7.utils.visualize import plot_tracking,plot_tracking_route
from yolov7.utils.timer import Timer
####### yolo deepdoc cut line ################
import Deepdoc.encoding.utils as utils
from Deepdoc.encoding.nn import BatchNorm2d
from Deepdoc.encoding.datasets import get_segmentation_dataset ,test_batchify_fn
from Deepdoc.encoding.models import get_model, get_segmentation_model ,MultiEvalModule

name_coco = [ 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
         'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
         'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
         'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
         'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
         'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
         'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
         'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
         'hair drier', 'toothbrush' ]

class Deeplab():
    def __init__(self,pretrained,opt,device,decode_fn):
        self.device = device
        self.decode_fn = decode_fn
        self.opt = opt
        self.pretrained = pretrained
        #print("55555555555555555555555555",self.opt.model)
        self.model = network.modeling.__dict__[self.opt.model](num_classes=21, output_stride=opt.output_stride)
       
        if self.opt.ckpt is not None and os.path.isfile(self.opt.ckpt):
            self.checkpoint = torch.load(self.opt.ckpt, map_location=torch.device('cpu'))
            self.model.load_state_dict(self.checkpoint["model_state"])
            self.model = nn.DataParallel(self.model)
            self.model.to(self.device)
            self.model.half()
            print("Resume model from %s" % self.opt.ckpt)
            #print(self.model.device)
            del self.checkpoint
        else:
            print("[!] Retrain")
            self.model = nn.DataParallel(self.model)
            self.model.to(self.device)
            #print(self.model.device)
        with torch.no_grad():
            self.model = self.model.eval()
        
    def detect(self,img_path):
        if self.opt.crop_val:
            transform = T.Compose([
                    T.Resize((224,224)),
                    T.Resize(opt.crop_size),
                    T.CenterCrop(opt.crop_size),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]),
                ])
        else:
            transform = T.Compose([
                T.Resize((224,224)),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]),
                ])
        
        ext = os.path.basename(img_path).split('.')[-1]
        img_name = os.path.basename(img_path)[:-len(ext)-1]
        img = Image.open(img_path).convert('RGB')
        (w,h) = img.size
        img = transform(img).unsqueeze(0) # To tensor of NCHW
        img = img.to(self.device)
        img = img.half()
        
        pred =self.model(img).max(1)[1].cpu().numpy()[0] # HW
        colorized_preds = self.decode_fn(pred).astype('uint8')
        colorized_preds = Image.fromarray(colorized_preds)
       
        
        return colorized_preds
            
class Yolov7_tracker():
    
    def __init__(self, opt,weights,device,imgsz,trace):
        self.device = device
        self.opt = opt
        self.yolo_model = attempt_load(weights,map_location=self.device)
        self.stride = int(self.yolo_model.stride.max()) #model stride
        self.imgsz = check_img_size(imgsz,s=self.stride)
        self.tracker = BYTETracker(opt,frame_rate=30)
        if trace:
            self.model = TracedModel(self.yolo_model , device, opt.img_size)

       
        self.names = model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names] 
        
    def predict(self,img):
        
        if self.device.type !="cpu":
            half = True
            self.model(torch.zeros(1,3,self.imgsz,self.imgsz).to(self.device).type_as(next(self.model.parameters())))
        old_img_w = old_img_h = self.imgsz
        old_img_b = 1
        
        
        img = torch.from_numpy(img).to(self.device)
        img =  img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        if self.device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                self.model(img, augment=self.opt.augment)[0]

        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = self.model(img, augment=self.opt.augment)[0]

        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()
        img = 255*(img.cpu())
        return pred,img ,t1 ,t2 ,t3
     
        
        
    def track(self,pred,vid_cap,img,img_size):
        
        out = cv2.cvtColor(np.transpose(img[0].numpy().astype(np.uint8),(1,2,0)),cv2.COLOR_BGR2RGB)
        
        cls_list = []
        for i,det in enumerate(pred):
            
            for *xyxy, conf, cls in reversed(det):
                #print("xyxy",*xyxy)
                results = []
                online_cls = []
                online_tlwhs = []
                online_ids = []
                online_scores = []
                
                obj_conf = torch.ones(det.shape[0],1).to(self.device)
               
                det2 = torch.cat([det[ :,:4],det[ :,4:]],axis=1)
                det2=det2.cpu()
                online_targets =self.tracker.update(det2, [img_size[0],img_size[1]], img_size)
                
                
                
                for t in online_targets:
                   
                    tlwh = t.tlwh
                    track_cls = t.test_cls
                    tid = t.track_id
                    vertical = tlwh[2] / tlwh[3] > opt.aspect_ratio_thresh
                    if tlwh[2] * tlwh[3] > opt.min_box_area and not vertical:
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                        online_scores.append(t.score)
                        online_cls.append(int(track_cls))
               
            if vid_cap:  # video
                fps = vid_cap.get(cv2.CAP_PROP_FPS)
            else:
                fps = 0
            
        return out,online_tlwhs,online_ids,cls_list,fps,online_cls

class DeepDoc():
    def __init__(self,pretrained,opt,device):
        self.device = device
        self.opt = opt
        self.pretrained = pretrained
        self.model = get_segmentation_model(name = self.opt.deepdoc_mod,backbone=self.opt.backbone,dilated=self.opt.dilated,lateral = self.opt.lateral, 
                                    jpu = self.opt.jpu, aux = self.opt.aux,
                                    se_loss = self.opt.se_loss, norm_layer = BatchNorm2d,
                                    base_size = self.opt.base_size, crop_size = self.opt.crop)

        if self.opt.resume is None or not os.path.isfile(self.opt.resume):
            raise RuntumeError("=> no checkpoint found at '{}'" .format(self.opt.resume))
        self.checkpoint = torch.load(self.opt.resume)
        self.model.load_state_dict(self.checkpoint["state_dict"])
        print('---------- Networks initialized -------------')
        self.scales = [0.5,1.0]
        self.evaluator = MultiEvalModule(self.model,8,scales=self.scales,flip=True)
        self.evaluator.to(self.device)
        with torch.no_grad():
            self.evaluator = self.evaluator.eval()
        self.metric = utils.SegmentationMetric(nclass=8)
        self.mapping = {
                    0: 0,  
                    1: 200, #uterus
                    2: 100, #ovary
                    3: 150, #uterus tube
                    4: 50,  #colon
                    5: 225, #ovarian tumor
                    6: 255, #tools
                    7: 250  #myoma
                }

    def mask_to_class(mask):
        for k in mapping:
            mask_copy = np.copy(mask)
            mask_copy[mask==k] = mapping[k]
        return mask_copy

    def predict(self,image,img_path):
        transform = transform.Compose([
            transform.ToTensor(),
            transform.Normalize([.485, .456, .406], [.229, .224, .225])])
        image = transform(image).unsqueeze(0)
        image = image.to(self.device)
        out = self.evaluator.parallel_forward(image)
        pred = [(torch.max(output, 1)[1].cpu().numpy())
                            for output in outputs]

        mask = Image.fromarray(pred.squeeze().astype('uint8'))
        mask = np.asarray(mask)
        mask = mask_to_class(mask)

        return mask

def translucent(bottom,top):
    
    out = cv2.addWeighted(bottom,1,top,0.5,0)
    
    return out
    
def basic(input_video = 'outdoor.mp4', source = "LASAD-YOLO-s", output_video = 'out.mp4'):
    video = cv2.VideoCapture(input_video) # 23.97602537435857
    fps = video.get(cv2.CAP_PROP_FPS)
    print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))

    img = cv2.imread(os.path.join(source, '1.png'))  
    size = (img.shape[1],img.shape[0])  
    print(size)
    fourcc = cv2.VideoWriter_fourcc(*"XVID") 
    videoWrite = cv2.VideoWriter(output_video,fourcc,fps,size)

    files = os.listdir(os.path.join(source))
    out_num = len(files)
    for i in range(1, out_num):
        
        fileName = os.path.join(source, f'{i}.png')   
        img = cv2.imread(fileName)
        videoWrite.write(img)
    videoWrite.release()    
        
    
            
def main(opt):
    # setup dataloader
    
    source,weights,view_img,save_txt,imgsz,trace,yolo_dataset=  opt.source,opt.weights,opt.view_img,opt.save_txt,opt.img_size,not opt.no_trace,opt.yolo_dataset
    
    if opt.dataset.lower() == 'voc':
        opt.num_classes = 21
        decode_fn = VOCSegmentation.decode_target
    elif opt.dataset.lower() == 'cityscapes':
        opt.num_classes = 19
        decode_fn = Cityscapes.decode_target
        
    ave_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    device1 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device2 = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    if opt.seg_mod == "DeepDoc":
        deepdoc = DeepDoc(opt.resume,opt,device1)
        print("Deepdoc on!")
    if opt.seg_mod == "deeplab":
        deeplab = Deeplab(opt.ckpt,opt,device1,decode_fn)
        print("deeplab on!")
    
    yolov7_track = Yolov7_tracker(opt,weights,device2,imgsz,trace)
    dataset = LoadImages(source,img_size=yolov7_track.imgsz,stride=yolov7_track.stride)
    
    try:
        os.makedirs(f"result/{source}")
        os.makedirs(f"mask/{source}")
    except FileExistsError:
        pass
        
    count = 0
    for path, img, im0s, vid_cap in tqdm(dataset):
        
        c,h,w= img.shape
        if vid_cap:
            fps = vid_cap.get(cv2.CAP_PROP_FPS)
        if opt.seg_mod =="deeplab":
            segment_out = deeplab.detect(path)
            segment_out =cv2.cvtColor(np.asarray(segment_out), cv2.COLOR_RGB2BGR)
            
            segment_out = cv2.resize(segment_out,(w,h),interpolation=cv2.INTER_LINEAR)
            
            cv2.imwrite(os.path.join(f"mask/{source}",f"{count}.png"),segment_out)
            image = torch.from_numpy(img)
            image = image[np.newaxis, :]
            image = image.permute(2,3,1,0)
            image = torch.squeeze(image)
       
        seg_img_fuse = translucent(image.numpy(),segment_out)
        seg_img_fuse = cv2.cvtColor(np.asarray(seg_img_fuse), cv2.COLOR_BGR2RGB)
        
        yolo_out,img_cpu,_,_,_ = yolov7_track.predict(img)
        try:
            out,online_tlwhs,online_ids,cls_list,fps,track_cls = yolov7_track.track(yolo_out,vid_cap,img_cpu,(w,h))
            translucent_track = plot_tracking_route(seg_img_fuse, online_tlwhs,track_cls,online_ids, frame_id=count+1, fps=fps,yolo_dataset=yolo_dataset)
        except UnboundLocalError:
             translucent_track =seg_img_fuse

        
        count +=1
        
        #print(translucent_track)
        save_path = os.path.join(f"result/{source}")
        cv2.imwrite(os.path.join(save_path,f"{count}.png"),translucent_track)
        
    basic(input_video = opt.input_video, source = save_path, output_video = 'out.mp4')
            
    
if __name__ == '__main__':
    
    # yolo options
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')
    parser.add_argument('--input_video', type=str, default='videos/bus.mp4', help='source')
    # file/folder, 0 for webcam
    parser.add_argument('--yolo_dataset', type=str, default='davinci')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.7, help='object confidence threshold')
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

    parser.add_argument("--model", type=str, default= 'deeplabv3plus_effiformer',
                        choices=available_models,help='model name')
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
    parser.add_argument("--crop_size", type=int, default=224)

    
    parser.add_argument("--ckpt", default=None, type=str,
                        help="resume from checkpoint")
    ### Deepdoc options ###
    ## model / backbone / resume --mode  --aux --se-loss --jpu
    parser.add_argument('--deepdoc_mod', type=str, default='encnet',
                            help='model name (default: encnet)')
    parser.add_argument('--backbone', type=str, default='efficientnet',
                        help='backbone name (default: resnet50)')
    parser.add_argument('--jpu', action='store_true', default=
                        False, help='JPU')
    parser.add_argument('--workers', type=int, default=16,
                            metavar='N', help='dataloader threads')
    parser.add_argument('--base_size', type=int, default=520,
                        help='base image size')
    parser.add_argument('--crop', type=int, default=480,
                        help='crop image size')
    parser.add_argument('--aux', action='store_true', default= False,
                            help='Auxilary Loss')
    parser.add_argument('--aux-weight', type=float, default=0.2,
                        help='Auxilary loss weight (default: 0.2)')
    parser.add_argument('--se-loss', action='store_true', default= False,
                        help='Semantic Encoding Loss SE-loss')
    parser.add_argument('--se-weight', type=float, default=0.2,
                        help='SE-loss weight (default: 0.2)')
    parser.add_argument('--no-cuda', action='store_true', default=
                    False, help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--dilated', action='store_true', default=
                            False, help='dilation')


    parser.add_argument('--lateral', action='store_true', default=
                    False, help='employ FPN')
    # # # checking point
    parser.add_argument('--resume', type=str, default=None,
                            help='put the path to resuming file if needed')
    ##### segment model option ############
    parser.add_argument("--seg_mod",type=str,default="deeplab",
                            help = "choose which segment model you need")
    ########### option endline ######################
    
    opt = parser.parse_args()
    main(opt)