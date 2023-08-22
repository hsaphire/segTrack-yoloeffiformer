#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import cv2
import numpy as np
import os
__all__ = ["vis"]

track_route = []
def vis(img, boxes, scores, cls_ids, conf=0.5, class_names=None):

    for i in range(len(boxes)):
        box = boxes[i]
        cls_id = int(cls_ids[i])
        score = scores[i]
        if score < conf:
            continue
        x0 = int(box[0])
        y0 = int(box[1])
        x1 = int(box[2])
        y1 = int(box[3])

        color = (_COLORS[cls_id] * 255).astype(np.uint8).tolist()
        text = '{}:{:.1f}%'.format(class_names[cls_id], score * 100)
        txt_color = (0, 0, 0) if np.mean(_COLORS[cls_id]) > 0.5 else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX

        txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)

        txt_bk_color = (_COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
        cv2.rectangle(
            img,
            (x0, y0 + 1),
            (x0 + txt_size[0] + 1, y0 + int(1.5*txt_size[1])),
            txt_bk_color,
            -1
        )
        cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1)

    return text,x0,y0


def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)

    return color

names=[ "Fenastrate",
    "Spetula",
    "Scissors",
    "Suction&irrigation",
    "Tenaculum",
    "Bipolar focept",
    "Needle Driver" ]
name_coco = [str(i) for i in range(80)]
def plot_tracking(image, tlwhs,obj_ids, scores=None, frame_id=0, fps=0., ids2=None):
    im = np.ascontiguousarray(np.copy(image)).astype(np.uint8)
    im_h, im_w = im.shape[:2]

    top_view = np.zeros([im_w, im_w, 3], dtype=np.uint8) + 255

    #text_scale = max(1, image.shape[1] / 1600.)
    #text_thickness = 2
    #line_thickness = max(1, int(image.shape[1] / 500.))
    text_scale = 2
    text_thickness = 2
    line_thickness = 3
    
    radius = max(5, int(im_w/140.))
    cv2.putText(im, 'frame: %d fps: %.2f num: %d' % (frame_id, fps, len(tlwhs)),
                (0, int(15 * text_scale)), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), thickness=2)
    
    for i, tlwh in enumerate(tlwhs):
        x1, y1, w, h = tlwh
        
        center_point = [x1+w/2,y1+h/2]
        intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
        #print(intbox)
        track_route.append(center_point)
        if len(track_route) > 30:
            track_route.pop(0)
        obj_id = int(obj_ids[i])
        id_text = '{}'.format(int(obj_id))
        if ids2 is not None:
            id_text = id_text + ', {}'.format(int(ids2[i]))+ name[int(obj_cls)]
        color = get_color(abs(obj_id))
        if track_route:
            #print(len(track_route))
            for route in track_route:
                #print(route[0],route[1])
                cv2.circle(im,(int(route[0]),int(route[1])),1,[0,0,255],1)
        if intbox:
            #print(im.shape)
           # print(intbox[0:4])
            #print(id_text)
            cv2.rectangle(im, intbox[0:2], intbox[2:4], color=color, thickness=1)
            cv2.putText(im, id_text,(intbox[0], intbox[1]), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255),
                        thickness=2)
        else:
            print("err")
        #print("546456446",im.shape)
        #cv2.imwrite("5546456546456456456456.jpg",im)
        #assert False
    return im

obj_point_list = {}
def path(obj_route,h,w,vis_id):   #here need to insert id that you want
    background = np.zeros([h,w,3])

    for point in obj_route[vis_id]:
        cv2.circle(background,(point),2,[255,255,255],-1)
    return background


def plot_tracking_route(image, tlwhs,obj_clses,obj_ids, scores=None, frame_id=0, fps=0., ids2=None,source=None):
    im = np.ascontiguousarray(np.copy(image)).astype(np.uint8)
    im_h, im_w = im.shape[:2]
    
    top_view = np.zeros([im_w, im_w, 3], dtype=np.uint8)
    #centerpoints=[]
    #track_image_dir = os.path.join('img_save',"track_routes")
    #text_scale = max(1, image.shape[1] / 1600.)
    #text_thickness = 2
    #line_thickness = max(1, int(image.shape[1] / 500.))
    text_scale = 0.5
    text_thickness = 2
    line_thickness = 3
    
    for i, (tlwh,obj_cls) in enumerate(zip(tlwhs,obj_clses)):
        track_black = np.zeros([im_h,im_w,3])
        x1, y1, w, h = tlwh
        
        center_point = [x1+w/2,y1+h/2]
        intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
        obj_id = int(obj_ids[i])
        
        """
        2023/8/9
        
        """
        ########################################################################
        if obj_id in obj_point_list :
            obj_point_list[obj_id].append((int(center_point[0]),int(center_point[1]))) 
        else :
            obj_point_list[obj_id] = [(int(center_point[0]),int(center_point[1]))] 
        #print("obj_point_list",len(obj_point_list))    
        ########################################################################
        
        try:
            os.makedirs(f"img_save/{source}/track_routes/{obj_id}")
        except FileExistsError:
            pass
        track_image_dir= os.path.join(f"img_save/{source}/track_routes/{obj_id}")
        
        background = path(obj_point_list,im_h,im_w,obj_id)
        save_id = 1
        while True:
            save_route_png = f"/{save_id}.png"
            
            if os.path.exists(track_image_dir+save_route_png):               
                save_id +=1
                pass
            else:
                cv2.imwrite(track_image_dir+save_route_png,background)
                break
        
        track_route.append(center_point)
        if len(track_route) > 30:
            track_route.pop(0)
        

        id_text = '{}'.format(int(obj_id))
        if ids2 is not None:
            id_text = id_text + ', {}'.format(int(ids2[i]))
        color = get_color(abs(obj_id))
        if track_route:
            #print(len(track_route))
            for route in track_route:
                #target track_id : obj_id
                
                cv2.circle(im,(int(route[0]),int(route[1])),1,[0,0,255],1)
         
        if intbox:
            #print("obj_cls",obj_cls)
            lines = id_text +" "+names[obj_cls]
            #lines = id_text +" "+name_coco[obj_cls]
            
            #print(type(lines))
            #print(lines,intbox[0:2])
            #print("aaaaaaaaaaaaaaaaaaa",id_text)
            cv2.rectangle(im, intbox[0:2], intbox[2:4], color=color, thickness=1)
            cv2.putText(im, lines, (intbox[0], intbox[1]), 0, text_scale, (0, 0, 255),
                        thickness=1,lineType=cv2.LINE_AA)
            #cv2.imwrite(f"{i}_img.png", im)
            #print(i)
        else:
            print("err")
        #print("546456446",im.shape)
        #cv2.imwrite("5546456546456456456456.jpg",im)
        #assert False
    return im

_COLORS = np.array(
    [
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        0.300, 0.300, 0.300,
        0.600, 0.600, 0.600,
        1.000, 0.000, 0.000,
        1.000, 0.500, 0.000,
        0.749, 0.749, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 1.000,
        0.667, 0.000, 1.000,
        0.333, 0.333, 0.000,
        0.333, 0.667, 0.000,
        0.333, 1.000, 0.000,
        0.667, 0.333, 0.000,
        0.667, 0.667, 0.000,
        0.667, 1.000, 0.000,
        1.000, 0.333, 0.000,
        1.000, 0.667, 0.000,
        1.000, 1.000, 0.000,
        0.000, 0.333, 0.500,
        0.000, 0.667, 0.500,
        0.000, 1.000, 0.500,
        0.333, 0.000, 0.500,
        0.333, 0.333, 0.500,
        0.333, 0.667, 0.500,
        0.333, 1.000, 0.500,
        0.667, 0.000, 0.500,
        0.667, 0.333, 0.500,
        0.667, 0.667, 0.500,
        0.667, 1.000, 0.500,
        1.000, 0.000, 0.500,
        1.000, 0.333, 0.500,
        1.000, 0.667, 0.500,
        1.000, 1.000, 0.500,
        0.000, 0.333, 1.000,
        0.000, 0.667, 1.000,
        0.000, 1.000, 1.000,
        0.333, 0.000, 1.000,
        0.333, 0.333, 1.000,
        0.333, 0.667, 1.000,
        0.333, 1.000, 1.000,
        0.667, 0.000, 1.000,
        0.667, 0.333, 1.000,
        0.667, 0.667, 1.000,
        0.667, 1.000, 1.000,
        1.000, 0.000, 1.000,
        1.000, 0.333, 1.000,
        1.000, 0.667, 1.000,
        0.333, 0.000, 0.000,
        0.500, 0.000, 0.000,
        0.667, 0.000, 0.000,
        0.833, 0.000, 0.000,
        1.000, 0.000, 0.000,
        0.000, 0.167, 0.000,
        0.000, 0.333, 0.000,
        0.000, 0.500, 0.000,
        0.000, 0.667, 0.000,
        0.000, 0.833, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 0.167,
        0.000, 0.000, 0.333,
        0.000, 0.000, 0.500,
        0.000, 0.000, 0.667,
        0.000, 0.000, 0.833,
        0.000, 0.000, 1.000,
        0.000, 0.000, 0.000,
        0.143, 0.143, 0.143,
        0.286, 0.286, 0.286,
        0.429, 0.429, 0.429,
        0.571, 0.571, 0.571,
        0.714, 0.714, 0.714,
        0.857, 0.857, 0.857,
        0.000, 0.447, 0.741,
        0.314, 0.717, 0.741,
        0.50, 0.5, 0
    ]
).astype(np.float32).reshape(-1, 3)
