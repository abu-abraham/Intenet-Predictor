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
        self._x = 0
        self._y = 0
        self._counter = 0
        self.identifier_last_update_frame_map = dict()
        self.identifier_coordinate_map = dict()
        self.identifier_action_map = dict()
        self.identifier_image_queue_map = dict()
        self.action_recognizer = ActionRecognizer()
        self.action_recognizer_execution_thread = ParallelThread(self.getComputedAction)
        self.action_recognizer_execution_thread.start()
        self.net = build_ssd('test', 300, 21)
        self.net.load_weights('/home/abu/Documents/Personal/ImageIdentifier/Intent-Predictor/server/object_detection/weights/ssd300_mAP_77.43_v2.pth')
        self.object_color_map = {}
        self.action_color_map = {}
        self._16x16 = np.zeros(shape=(16,16))
        self.detected_objects = {}
        self.lstm_queue = []
        self.object_wise_split = {}
        self.earlier_seen = {}

    def start(self):
        print("Object detector initialized!")

    def generateUniqueIdentifier(self, coordinates):
        return ((coordinates[0][0]+coordinates[0][1])/(coordinates[1]+coordinates[2]))

    def getSubImage(self, coords, image):
        images = []
        #800 is the assumed frame size! should be increased
        diff = 400-abs(int(coords[0][1])-int(coords[0][1]+coords[2]))
        diff = int(diff/2)
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
        return images


    def saveImage(self, image):
        image.save('filler'+str(self.x)+'.png')

    def validDifference(self, cord, coordinates, earlier_coordinates):
        limit = 70
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
                #detected_action = self.action_recognizer.identifyAction(copy.deepcopy(self.identifier_image_queue_map[u_id]))
                self.identifier_action_map[u_id] = "running"#detected_action
                self.identifier_image_queue_map[u_id] = []

    def formattedImage(self, image, name = "saved1.png"):
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
            x_p = x_p[:400,:400]
            img_arr[i,:,:] = x_p

        image_np = Image.fromarray(np.uint8(img_arr).transpose(1,2,0))
        im = Image.fromarray(np.uint8(image_np))
        im.save(name)
        return image_np


    def detect_objects(self, image):
        self._x = image.shape[1]
        self._y = image.shape[0]
        print("Image shape", image.shape)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        x = cv2.resize(image, (300, 300)).astype(np.float32)
        #x -= (104.0, 117.0, 123.0)
        x = x.astype(np.float32)
        x = x[:, :, ::-1].copy()
        x = torch.from_numpy(x).permute(2, 0, 1)

        xx = Variable(x.unsqueeze(0))
        # if torch.cuda.is_available():
        #     xx = xx.cuda()
        y = self.net(xx)

        detections = y.data
        scale = torch.Tensor(rgb_image.shape[1::-1]).repeat(2)
        detected = {}
        detected_scores = {}
        for i in range(detections.size(1)):
            j = 0
            while detections[0,i,j,0] >= 0.6 or (detections[0,i,j,0] >= 0.1 and labels[i-1] in self.detected_objects and self.isSameObject(labels[i-1],(detections[0,i,j,1:]*scale).cpu().numpy())):
                label_name = labels[i-1]
                pt = (detections[0,i,j,1:]*scale).cpu().numpy()
                coords = (pt[0], pt[1]), pt[2]-pt[0]+1, pt[3]-pt[1]+1
                if label_name!="person":
                    detected_scores[label_name] = detections[0,i,j,0].item()
                detected.setdefault(label_name,[]).append([self.getSubImage(coords, image),coords])
                self.detected_objects[label_name] = coords
                j+=1

        ##Option 1: if item was detected in last 10 frames, add it here as well!

        for key in self.earlier_seen:
            if key !="person":
                detected.setdefault(key,[]).append(self.earlier_seen[key]) 

        for key in detected:
            res = {v:k for k,v in detected_scores.items()}
            res = next(iter(res.values()))
            #We are just focused on uniquely identifying people as only human action is recognized now.
            if key != "person":
                # for entry in detected[key]:
                #     self.insertTo16x16(key, entry[1])
                continue
            for entry in detected[key]:
                print(key, entry[1])
                u_id = self.getMappedIdentifier(entry[1])
                if u_id==None:
                    u_id = self.generateUniqueIdentifier(entry[1])
                    self.identifier_coordinate_map[u_id] = entry[1]
                self.identifier_image_queue_map.setdefault(u_id,[]).append(self.formattedImage(entry[0]))
                self.insertToNew("person", entry[1], res, detected[res][0][1] )
                self.insertTo16x16(key, entry[1])
        self.earlier_seen = detected
        self.nextStep()
        self.save_16x16_as_216x216()
        self._16x16 = np.zeros(shape=(16,16))

    def computeDistance(self, x1,y1,x2,y2):
        import math
        return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)


    def insertToNew(self, *args):
        arg = list(args)
        dist = (self.computeDistance((arg[1][0][0]+arg[1][1])/2,(arg[1][0][1]+arg[1][2])/2,(arg[3][0][0]+arg[3][1])/2,(arg[3][0][1]+arg[3][2])/2))/1000
        if(len(self.lstm_queue)) < 90:
            self.object_wise_split.setdefault(('P',self.getIndexedValue(arg[2],None)),[]).append([4,self.getIndexedValue(arg[2],None),dist])
            self.lstm_queue.append([4,self.getIndexedValue(arg[2],None),dist])
        else:
            self.object_wise_split.setdefault(('P',self.getIndexedValue(arg[2],None)),[]).append([5,self.getIndexedValue(arg[2],None),dist])
            self.lstm_queue.append([5,self.getIndexedValue(arg[2],None),0])


    def segmentedImage(self, image):
        return

    def isSameObject(self, label, pt):
        coords = (pt[0], pt[1]), pt[2]-pt[0]+1, pt[3]-pt[1]+1
        return self.validDifference(0, coords, self.detected_objects[label]) and self.validDifference(1, coords, self.detected_objects[label])


    def getReducedCoordinate(self, coordinates):
        return int(15*(coordinates[0][0]+(coordinates[1]/2))/self._x), int(15*(coordinates[0][1]+(coordinates[2]/2))/self._y)

    def nextStep(self):
        ##pass and train LSTM model with this 16x16 vector  (self.identifier_action_map)
        print("Next step")

    def getIndexedValue(self, key, coordinates):
        if key == 'person':
            return
            u_id = self.getMappedIdentifier(coordinates)
            if u_id not in self.identifier_action_map:
                return 1000-999
            action = self.identifier_action_map[u_id]-1000+1
            if action in self.action_color_map:
                return self.action_color_map.get(action)
            else:
                self.action_color_map[action] = (len(self.action_color_map)+1)*1000
                return self.action_color_map.get(action)

        else:
            if key in self.object_color_map:
                return self.object_color_map.get(key)
            else:
                self.object_color_map[key] = len(self.object_color_map)+1
                return self.object_color_map.get(key)

            

    def insertTo16x16(self, key, coordinates):
        x, y = self.getReducedCoordinate(coordinates)
        #print("X,y",x,y)
        if self._16x16[x][y]!=0:
            x,y = self.nextFreeSpot(x,y)
        value = self.getIndexedValue(key, coordinates)
        self._16x16[x][y] = value

    def nextFreeSpot(self, x, y):
        for i in range(max(0,x-1),min(15,x+1)):
            for j in range(max(0,y-1),min(15,y+1)):
                if self._16x16[i][j]==0:
                    return i,j


    def save_16x16_as_216x216(self):
        if False:
            temp = self._16x16.reshape((256))
            self.lstm_queue.append(temp)
            if len(self.lstm_queue) > 110:
                data = np.array(self.lstm_queue)
                torch.save(data, open('traindata.pt', 'wb'))
                self.lstm_queue = []
            print("Frame size, ",len(self.lstm_queue))
            return
        if True:
            if len(self.lstm_queue) > 110:
                print("Creating train data")
                data = np.array(self.lstm_queue)
                torch.save(data, open('traindata.pt', 'wb'))
                self.lstm_queue = []
                for entry in self.object_wise_split:
                    data = self.object_wise_split[entry]
                    data = np.array(data)
                    torch.save(data, open('traindata'+str(entry[1])+'.pt','wb'))
            else:
                print(len(self.lstm_queue))
            return

        A = np.full((216,216),250)
        A = np.stack((A,)*3, -1)
        for i in range(16):
            for j in range(16):
                value = self._16x16[i][j]
                for i1 in range(16*i,16*(i+1)):
                    for j1 in range(16*i,16*(i+1)):
                        if value!=0 and value < 1000:
                            A[i1][j1]=[value,value*10,(value*1000)%255]
                        elif value!=0:
                            A[i1][j1] = [value%255,value%100,value%160]
        im = Image.fromarray(A.astype('uint8'))
        im.save('output/'+str(self._counter)+'.jpeg')
        self._counter+=1

        

    


        




