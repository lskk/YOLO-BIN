import os
import cv2
import sys
from darkflow.net.build import TFNet
#from finding_lane_1 import detect_lanes_img
import numpy as np
#import serial
import time
import json
import requests
import logging


# watchdogs dir listener
from os import listdir, remove, rename
from os.path import isfile, join, abspath, dirname, exists
from watchdog.events import PatternMatchingEventHandler
from watchdog.observers import Observer

logging.basicConfig(level=logging.ERROR)
folder_path = sys.argv[1]
options = {
    'model': 'cfg/yolo-voc.cfg',
    'load': 'bin/yolo-voc.weights',
    'threshold': 0.3,
    'gpu': 0.5
}


def draw_boxes(colors, results, frame):
    #arduino = serial.Serial('COM3',9600)
    json_file = []
    for (color, result) in zip(colors, results):
        # Convert confidence level to int from float
        json_temp = dict(result)
        json_conf = json_temp['confidence']
        json_conf = int(round(json_conf*100))
        json_temp['confidence'] = json_conf
        # till here
        json_file.append(json_temp)
        tl = (result['topleft']['x'], result['topleft']['y'])
        br = (result['bottomright']['x'], result['bottomright']['y'])
        confidence = result['confidence']
        label = result['label']
        coorX = result['topleft']['x']+result['bottomright']['x']/2
        coorY = result['topleft']['y']+result['bottomright']['y']/2
        centroids = (coorX, coorY)
        #print('{},{},{},{}\n'.format(result['topleft']['x'], result['topleft']['y'],result['bottomright']['x'], result['bottomright']['y']))
        text = '{}: {:.1f}%'.format(label, confidence*100)
        frame = cv2.rectangle(frame, tl, br, color, 5)
        frame = cv2.putText(
            frame, text, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
        # API
        # r = requests.post("http://167.205.7.227:8070/api/getFacebookData",json={'label' : json_file})
    cv2.imshow("predicted", frame)
    with open('data.json', 'w') as outfile:
        json.dump(json_file, outfile)


class MyEventHandler(PatternMatchingEventHandler):
    """docstring for MyEventHandler"""
    # patterns = ["*.jpg"]  # image format

    def __init__(self, observer):
        super(MyEventHandler, self).__init__()
        self.observer = observer
        self.imgFiles = []

    def on_created(self, event):
        if not event.is_directory:
            print("created")
            self.yolo(event)

    def yolo(self, event):
        tfnet = TFNet(options)
        fileRead = input(event.src_path)
        fileExtension = os.path.splitext(event.src_path)
        print(event.src_path)
        fileInput = (fileRead)
        fileInput.split("\")
        go = fileInput.split("\")
        print(go)

         colors = [tuple(255 * np.random.rand(3)) for _ in range(10)]

         if (fileExtension == '.jpg'):
             frame = cv2.imread(fileRead)
             results = tfnet.return_predict(frame)
             draw_boxes(colors, results, frame)
             cv2.waitKey(0)
             cv2.destroyAllWindows()
         elif (fileExtension == '.mp4'):
             capture = cv2.VideoCapture(fileRead)
             capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
             capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
             while True:
                 stime = time.time()
                 ret, frame = capture.read()
                 results = tfnet.return_predict(frame)
                 if ret:
                     #frame = detect_lanes_img(frame)
                     draw_boxes(colors, results, frame)
                     cv2.imshow("predicted", frame)
                     print('FPS {:.1f}'.format(1 / (time.time() - stime)))
                    print('frame : {},{}\n'.format(frame.shape[0], frame.shape[1]))
                 if cv2.waitKey(1) & 0xFF == ord('q'):
                     break
             capture.release()
             cv2.destroyAllWindows()
         elif (fileExtension == '.0' or fileExtension == '.1'):
             fileExtension = fileExtension[-1]
             cam = int(fileExtension)
             capture = cv2.VideoCapture(cam)
             while True:
                 stime = time.time()
                 ret, frame = capture.read()
                 results = tfnet.return_predict(frame)
                 if ret:
                     #frame = detect_lanes_img(frame)
                     draw_boxes(colors, results, frame)
                     cv2.imshow("predicted", frame)

                     print('FPS {:.1f}'.format(1 / (time.time() - stime)))
                     print('frame : {}\n'.format(frame.shape[0]))
                 if cv2.waitKey(1) & 0xFF == ord('q'):
                     break
             capture.release()
             cv2.destroyAllWindows()
         else:
             print('Input Invalid')


# M A I N   P R O G R A M


def main(argv=None):
    path = argv[0]

    observer = Observer()
    event_handler = MyEventHandler(observer)

    observer.schedule(event_handler, path, recursive=False)
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()

    observer.join()

    return 0


if __name__ == "__main__":
    main([folder_path])
