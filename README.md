# SegTrack system on uterine tumor

## Part1 DeepLabv3Plus

### Segentationtation Result on Pascal2012

<div>
<img src="samples/1_image.png"   width="20%">
<img src="samples/1_target.png"  width="20%">
<img src="samples/1_pred.png"    width="20%">
<img src="samples/1_overlay.png" width="20%">
</div>

<div>
<img src="samples/23_image.png"   width="20%">
<img src="samples/23_target.png"  width="20%">
<img src="samples/23_pred.png"    width="20%">
<img src="samples/23_overlay.png" width="20%">
</div>

<div>
<img src="samples/114_image.png"   width="20%">
<img src="samples/114_target.png"  width="20%">
<img src="samples/114_pred.png"    width="20%">
<img src="samples/114_overlay.png" width="20%">
</div>

## Part2 YOLOv7

### Object Detection on Random image 

<div>
<img src="samples/horses_prediction.jpg" width="59%"/>    
</div>

### Muti-Object Tracking

<div>
<img src="samples/MOT17-07-SDP.gif" width="400">
<div>

## Part3 Dataset

### DaVinci datasets

<div>
<img src="samples/3 (1).jpg" width=50%>
<img src="samples/3(1)_mask.jpg" width=50%>
<div>

### Bash
```bash
python predict.py --seg_mod deeplab  --model deeplabv3plus_effiformer_s2 --ckpt pretrained/best_deeplabv3plus_effiformer_s2_voc_os8.pth --yolo_dataset davinci --weight pretrained/best.pt --source videos/video1 --crop_val
```

### Demo

<div>
<img src="samples/out10.gif" width=500>
<div>
