import os
import sys

module_path = os.path.abspath(os.path.join('..'))

if module_path not in sys.path:
    sys.path.append(module_path)

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import numpy as np
import cv2
from PIL import Image
from numpy import newaxis
import sys
from object_detection.ssd import build_ssd
from matplotlib import pyplot as plt
from object_detection.data import VOCDetection, VOC_ROOT, VOCAnnotationTransform
from object_detection.data import VOC_CLASSES as labels
from action_recognition.action_recognizer import ActionRecognizer
import threading
import time
import copy

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

class ParallelThread(threading.Thread):
    
    flag = True
    
    def __init__(self, callback):
        self.callback = callback
        threading.Thread.__init__(self)

    def run(self):
        while self.flag:
            self.callback()
            time.sleep(1)
    
    def stop(self):
        self.flag = False


 
class ObjectDetector:

    x = 0

    def __init__(self):
        print("Object Detector initialized")
        self.identifier_last_update_frame_map = dict()
        self.identifier_coordinate_map = dict()
        self.identifier_action_map = dict()
        self.identifier_image_queue_map = dict()
        self.action_recognizer = ActionRecognizer()
        self.action_recognizer_execution_thread = ParallelThread(self.getComputedAction)
        self.action_recognizer_execution_thread.start()
        self.net = build_ssd('test', 300, 21)
        self.net.load_weights('/home/abu/Documents/Personal/ImageIdentifier/Intent-Predictor/server/object_detection/weights/ssd300_mAP_77.43_v2.pth')
 
    

    def generateUniqueIdentifier(self, coordinates):
        return ((coordinates[0][0]+coordinates[0][1])/(coordinates[1]+coordinates[2]))

    def getSubImage(self, coords, image):
        images = []
        #800 is the assumed frame size! should be increased
        diff = 400-abs(int(coords[0][1])-int(coords[0][1]+coords[2]))
        diff = int(diff/2)
        print(diff)
        #diff = 0
        for i in range(int(coords[0][1])-diff,int(coords[0][1]+coords[2])+diff):
            row = []
            y_diff = 400-abs(int(coords[0][0])-int(coords[0][0]+coords[1]))
            y_diff = int(y_diff/2)
            #y_diff = 0
            for j in range(int(coords[0][0])-y_diff,int(coords[1]+coords[0][0]+y_diff)):
                row.append(image[i][j])
            images.append(row)
        images = np.asarray(images)
        print(images.shape)
        return images


    def saveImage(self, image):
        image.save('filler'+str(self.x)+'.png')

    def validDifference(self, cord, coordinates, earlier_coordinates):
        limit = 25
        if abs(coordinates[0][cord]-earlier_coordinates[0][cord])<limit and abs(coordinates[cord+1]-earlier_coordinates[cord+1])< limit:
            return True
        return False

    def getMappedIdentifier(self, coordinates):
        for key in self.identifier_coordinate_map.keys():
            earlier_coordinates = self.identifier_coordinate_map[key]
            if self.validDifference(0, coordinates, earlier_coordinates) and self.validDifference(1, coordinates, earlier_coordinates):
                self.identifier_coordinate_map[key]= coordinates
                return key
        return None

    def getComputedAction(self):
        for u_id in self.identifier_image_queue_map:
            if (len(self.identifier_image_queue_map[u_id])>4):
                detected_action = self.action_recognizer.identifyAction(copy.deepcopy(self.identifier_image_queue_map[u_id]))
                self.identifier_action_map[u_id] = detected_action
                self.identifier_image_queue_map[u_id] = [];

    def formattedImage(self, image):
        image = image.transpose(2,0,1)
        leftPad = max(round(float((400 - image.shape[1])) / 2),0)
        rightPad = max(round(float(400 - image.shape[1]) - leftPad),0)
        topPad = max(round(float((400 - image.shape[2])) / 2),0)
        bottomPad = max(round(float(400 - image.shape[2]) - topPad),0)
        pads = ((leftPad,rightPad),(topPad,bottomPad))
        img_arr = np.ndarray((3,400,400),np.int)
        for i,x in enumerate(image):
            cons = np.int(np.median(x))
            x_p = np.pad(x,pads,'constant',constant_values=0)
            img_arr[i,:,:] = x_p
        image_np = Image.fromarray(np.uint8(img_arr).transpose(1,2,0))
        return image_np


    def detect_objects(self, image):
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        x = cv2.resize(image, (300, 300)).astype(np.float32)
        #x -= (104.0, 117.0, 123.0)
        x = x.astype(np.float32)
        x = x[:, :, ::-1].copy()
        x = torch.from_numpy(x).permute(2, 0, 1)

        xx = Variable(x.unsqueeze(0))
        if torch.cuda.is_available():
            xx = xx.cuda()
        y = self.net(xx)

        detections = y.data
        scale = torch.Tensor(rgb_image.shape[1::-1]).repeat(2)
        detected = {}
        for i in range(detections.size(1)):
            j = 0
            while detections[0,i,j,0] >= 0.6:
                label_name = labels[i-1]
                pt = (detections[0,i,j,1:]*scale).cpu().numpy()
                coords = (pt[0], pt[1]), pt[2]-pt[0]+1, pt[3]-pt[1]+1
                detected.setdefault(label_name,[]).append([self.getSubImage(coords, image),coords])
                print(coords)
                j+=1

        for key in detected:
            #We are just focused on uniquely identifying people as only human action is recognized now.
            if key != "person":
                continue
            for entry in detected[key]:
                u_id = self.getMappedIdentifier(entry[1])
                if u_id==None:
                    u_id = self.generateUniqueIdentifier(entry[1])
                    self.identifier_coordinate_map[u_id] = entry[1]
                self.identifier_image_queue_map.setdefault(u_id,[]).append(self.formattedImage(entry[0]))
        return self.identifier_action_map


