from socket import socket, AF_INET, SOCK_STREAM, SOL_SOCKET, SO_REUSEADDR
import _thread as thread
import time
import datetime
import numpy as np
from detection.object_detector import ObjectDetector

host = '0.0.0.0'
port = 1234
buf = 1024

addr = (host, port)

serversocket = socket(AF_INET, SOCK_STREAM)
serversocket.setsockopt(SOL_SOCKET, SO_REUSEADDR, 1)
serversocket.bind(addr)
serversocket.listen(10)


def handler(clientsocket, clientaddr):
    print("Accepted connection from: ", clientaddr)
    object_detector = ObjectDetector()
    while True:
        data = clientsocket.recv(1024).decode()
        array = np.array(data)
        objects_array = object_detector.detectObjects(array)
        clientsocket.send("Recieved Data".encode())
    clientsocket.close()



while True:
    try:
        print("Server is listening for connections\n")
        clientsocket, clientaddr = serversocket.accept()
        thread.start_new_thread(handler, (clientsocket, clientaddr))
    except KeyboardInterrupt:
        print("Closing server socket...")
        break

serversocket.close()
