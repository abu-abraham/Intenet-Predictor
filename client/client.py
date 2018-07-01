import cv2
import io
import socket
import struct
import time
import pickle
import zlib

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect(('localhost', 8485))
connection = client_socket.makefile('wb')

cam = cv2.VideoCapture('videos/video1.mp4')


img_counter = 0
encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
frames=int(cam.get(cv2.CAP_PROP_FRAME_COUNT))

while True and frames>0:
    ret, frame = cam.read()
    print(frame)
    result, frame = cv2.imencode('.jpg', frame, encode_param)
    data = pickle.dumps(frame, 0)
    size = len(data)
    client_socket.sendall(struct.pack(">L", size) + data)
    img_counter += 1
    frames-=1

cam.release()
