import argparse
import os, sys
import shutil
import time
from pathlib import Path

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

print(sys.path)
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import scipy.special
import numpy as np
import torchvision.transforms as transforms
import PIL.Image as image
from tracker import *
from lib.config import cfg
from lib.config import update_config
from lib.utils.utils import create_logger, select_device, time_synchronized
from lib.models import get_net
from lib.dataset import LoadImages, LoadStreams
from lib.core.general import non_max_suppression, scale_coords
from lib.utils import plot_one_box,show_seg_result
from lib.core.function import AverageMeter
from lib.core.postprocess import morphological_process, connect_lane
from tqdm import tqdm
normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])


modelConfiguration = 'yolov3-320.cfg'
modelWeigheights = 'yolov3-320.weights'
net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeigheights)
classesFile = "coco.names"
classNames = open(classesFile).read().strip().split('\n')
print(classNames)
# class index for our required detection classes
required_class_index = [2, 3, 5, 7,1,4]
detected_classNames = []
tracker = EuclideanDistTracker()
# Detection confidence threshold
confThreshold =0.2
nmsThreshold= 0.2

#  middle line and input size
input_size=320

colors = np.random.randint(0, 255, size=(len(classNames), 3), dtype='uint8')
def find_center(x, y, w, h):
    x1=int(w/2)
    y1=int(h/2)
    cx = x+x1
    cy=y+y1
    return cx, cy


def count_vehicle(box_id, img,speed):
    x, y, w, h, id = box_id
    # Find the center of the rectangle for detection
    center = find_center(x, y, w, h)
    ix, iy = center
    print(speed)
    if speed !=0:
         cv2.putText(img,str(speed)+"   ",(x+50,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,255,254),1)

    # cv2.putText(img,str(id)+" ",(x+50, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,245), 1)
    # Draw circle in the middle of the rectangle
    cv2.circle(img, center, 2, (0, 0, 255), -1) 

def detect(cfg,opt):

    logger, _, _ = create_logger(
        cfg, cfg.LOG_DIR, 'demo')

    device = select_device(logger,opt.device)
    if os.path.exists(opt.save_dir):  # output dir
        shutil.rmtree(opt.save_dir)  # delete dir
    os.makedirs(opt.save_dir)  # make new dir
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = get_net(cfg)
    checkpoint = torch.load(opt.weights, map_location= device)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)
    if half:
        model.half()  # to FP16

    # Set Dataloader
    if opt.source.isnumeric():
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(opt.source, img_size=opt.img_size)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(opt.source, img_size=opt.img_size)
        bs = 1  # batch_size


    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]


    # Run inference
    t0 = time.time()

    vid_path, vid_writer = None, None
    img = torch.zeros((1, 3, opt.img_size, opt.img_size), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    model.eval()

    inf_time = AverageMeter()
    nms_time = AverageMeter()
    
    for i, (path, img, img_det, vid_cap,shapes) in tqdm(enumerate(dataset),total = len(dataset)):
        img = transform(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        # Inference
        t1 = time_synchronized()
        det_out, da_seg_out,ll_seg_out= model(img)
        t2 = time_synchronized()
        # if i == 0:
        #     print(det_out)
        inf_out, _ = det_out
        inf_time.update(t2-t1,img.size(0))
        _, _, height, width = img.shape
        h,w,_=img_det.shape
        pad_w, pad_h = shapes[1][1]
        pad_w = int(pad_w)
        pad_h = int(pad_h)
        ratio = shapes[1][0][1]

        da_predict = da_seg_out[:, :, pad_h:(height-pad_h),pad_w:(width-pad_w)]
        da_seg_mask = torch.nn.functional.interpolate(da_predict, scale_factor=int(1/ratio), mode='bilinear')
        _, da_seg_mask = torch.max(da_seg_mask, 1)
        da_seg_mask = da_seg_mask.int().squeeze().cpu().numpy()
        # da_seg_mask = morphological_process(da_seg_mask, kernel_size=7)

        
        ll_predict = ll_seg_out[:, :,pad_h:(height-pad_h),pad_w:(width-pad_w)]
        ll_seg_mask = torch.nn.functional.interpolate(ll_predict, scale_factor=int(1/ratio), mode='bilinear')
        _, ll_seg_mask = torch.max(ll_seg_mask, 1)
        ll_seg_mask = ll_seg_mask.int().squeeze().cpu().numpy()
        # Lane line post-processing
        #ll_seg_mask = morphological_process(ll_seg_mask, kernel_size=7, func_type=cv2.MORPH_OPEN)
        #ll_seg_mask = connect_lane(ll_seg_mask)

        img_det = show_seg_result(img_det, (da_seg_mask, ll_seg_mask), _, _, is_demo=True) 


        frame = img_det
        ih, iw, channels = frame.shape
   
        blob = cv2.dnn.blobFromImage(frame, 1 / 255, (input_size, input_size), [0, 0, 0], 1, crop=False)
    # Set the input of the network
        net.setInput(blob)
        layersNames = net.getLayerNames()
        outputNames = [(layersNames[i - 1]) for i in net.getUnconnectedOutLayers()]
    # Feed data to the network
        outputs = net.forward(outputNames)
        height,width=frame.shape[:2]
        boxes=[]
        classIds=[]
        confidence_scores=[]
        detection=[]
        for output in outputs:
            for det in output:
                scores=det[5:]
                classId=np.argmax(scores)
                confidence=scores[classId]
                if classId in required_class_index:
                    if confidence > confThreshold:
                        w,h = int(det[2]*width) , int(det[3]*height)
                        x,y = int((det[0]*width)-w/2) , int((det[1]*height)-h/2)
                        boxes.append([x,y,w,h])
                        classIds.append(classId)
                        confidence_scores.append(float(confidence))
        indices=cv2.dnn.NMSBoxes(boxes,confidence_scores,confThreshold,nmsThreshold)
        if len(indices) > 0:
                            for i in indices.flatten():
                                x, y, w, h = boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]
                                # # color = "red"
                                
                                name = classNames[classIds[i]]
                                # detected_classNames.append(0)
                                cv2.putText(img_det,f'{name.upper()}',(x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0, 255, 0), 1)
                                # # # Draw bounding rectangle
                                cv2.rectangle(img_det, (x, y), (x + w, y + h),(0, 255, 255), 2)
                                # # #distance
                                
                                # # cv2.imshow('frame',img_det)
                                distance = ((2 * 3.14 * 180) / (w+h * 360) * 1000 + 3 )*0.254
                                cv2.putText(img_det,str("{:.2f} m".format(distance)),(x+50,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0, 255, 214),2)
                                # detection.append([x, y, w, h, required_class_index.index(classIds[i])])
 
        boxes_ids,speed = tracker.update(detection)

    
    # cv2.putText(frame, "count : "+str(len(boxes_ids)), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,254),2)
    
        # Apply NMS
        t3 = time_synchronized()
        det_pred = non_max_suppression(inf_out, conf_thres=opt.conf_thres, iou_thres=opt.iou_thres, classes=None, agnostic=False)
        t4 = time_synchronized()

        nms_time.update(t4-t3,img.size(0))
        det=det_pred[0]
 

        save_path = str(opt.save_dir +'/'+ Path(path).name) if dataset.mode != 'stream' else str(opt.save_dir + '/' + "web.mp4")

     
        if dataset.mode == 'images':
            cv2.imwrite(save_path,img_det)

        elif dataset.mode == 'video':
            if vid_path != save_path:  # new video
                vid_path = save_path
                if isinstance(vid_writer, cv2.VideoWriter):
                    vid_writer.release()  # release previous video writer

                fourcc = 'mp4v'  # output video codec
                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                h,w,_=img_det.shape
                vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
            vid_writer.write(img_det)
        
         # 1 millisecond

    print('Results saved to %s' % Path(opt.save_dir))
    print('Done. (%.3fs)' % (time.time() - t0))
    print('inf : (%.4fs/frame)   nms : (%.4fs/frame)' % (inf_time.avg,nms_time.avg))




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='weights/End-to-end.pth', help='model.pth path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder   ex:inference/images
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--save-dir', type=str, default='inference/output', help='directory to save results')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    opt = parser.parse_args()
    with torch.no_grad():
        detect(cfg,opt)
