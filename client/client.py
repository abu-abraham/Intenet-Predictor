import socket
import numpy as np
from PIL import Image
import cv2
import _thread as thread

host = '150.203.163.43/24'
host = 'localhost'
port = 1234
buf = 1024

clientsocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
clientsocket.connect((host, port))

cap = cv2.VideoCapture('videos/video1.mp4')
frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

while(True and frames>0):
    ret, frame = cap.read()
    clientsocket.send(np.asarray(frame).encode())
    print(clientsocket.recv(buf).decode())
    frames-=1

    
