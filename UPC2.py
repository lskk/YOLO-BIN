import os
import cv2

import sys
from darkflow.net.build import TFNet
# from finding_lane_1 import detect_lanes_img
import numpy as np
# import serial
import time
import json
import requests
import logging
import numpy as np

# saving image lib
# from PIL import Image
# watchdogs dir listener
from os import listdir, remove, rename
from os.path import isfile, join, abspath, dirname, exists
from watchdog.events import PatternMatchingEventHandler
from watchdog.observers import Observer

logging.basicConfig(level=logging.ERROR)
folder_path = sys.argv[1]
options = {
    'model': 'cfg/yolo-face.cfg',
    'load': 'bin/yolo-face_final.weights',
    'threshold': 0.3,
    'gpu': 0.8,
    'gpuName': '/gpu:0'
}

summary=[]
resultList=[]
resultFrame=[]
summaryList=[]
jsonTempList=[]
tfnet = TFNet(options)

def is_intersect(self, other):
    if self['topleft']['x'] > other['topleft']['x'] or self["bottomright"]["x"] < other["bottomright"]["x"]:
        return False
    if self['bottomright']['y'] > other['bottomright']['y'] or self['topleft']['y'] < other['topleft']['y']:
        return False
    return True


def draw_boxes(colors, results, frame, filePath):
    summary=[]
    jsonTempList=[]
  #  del summary[:]
   # del jsonTempList[:]
    for (color, result) in zip(colors, results):
       
        json_temp = dict(result)
        x1 = json_temp['topleft']['x']
        x2 = json_temp["bottomright"]["x"]
        y1 = json_temp['topleft']['y']
        y2 = json_temp['bottomright']['y']
        yuhulabel=json_temp['label']
        json_temp["warning"]=0
            
        if(any(xs["label"] == yuhulabel for xs in summary)):
            for xs in summary:
                    if(xs["label"]==yuhulabel):
                        xs["count"]+=1
        else:
            summary.append({"label": yuhulabel, "count": 1})

        
        
        json_temp['topright'] = {
            'x': json_temp["bottomright"]["x"], 'y': json_temp["topleft"]["y"]}
        json_temp['bottomleft'] = {
            'x': json_temp["topleft"]["x"], 'y': json_temp["bottomright"]["y"]}
        json_temp['center'] = {'x': ((x2-x1)/2)+x1, 'y': ((y2-y1)/2)+y1}
        json_conf = json_temp['confidence']
        json_conf = int(round(json_conf*100))
        json_temp['confidence'] = json_conf
        

                                   
                        
                                                
                            
        jsonTempList.append(json_temp)                    
        # till here
        
        tl = (result['topleft']['x'], result['topleft']['y'])
       # tr = (result['topright']['x'],result['topright']['y'])
        br = (result['bottomright']['x'], result['bottomright']['y'])
        confidence = result['confidence']
        label = result['label']
        text = '{}: {:.1f}%'.format(label, confidence*100)
        frame = cv2.rectangle(frame, tl, br, color, 5)
        frame = cv2.putText(
            frame, text, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
    
    #print("JSONTEMP")
   # print(jsonTempList)
    if summary:
        summaryList.append(summary)
        #print(summary)
    else:
        summaryList.append([])
    
    if jsonTempList:
        resultList.append(jsonTempList)
       # print(summary)
    else:
        resultList.append([])
    
    #if jsonTempList:
        #resultList.append(jsonTempList)
    #print(summaryList)
        
     
    


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

    def on_moved(self, event):
        if not event.is_directory:
            print("moved")
            self.yolo(event)
    def yolo(self, event):
        
        fileRead = (event.src_path)  # read src
        print(fileRead)
        spliteGo = os.path.basename(fileRead)
        print(spliteGo)
        fileExtension = os.path.splitext(spliteGo)
        print(fileExtension[0])
        if (fileExtension[1] == '.json'):
             print("Start Proccess")
             colors = [tuple(255 * np.random.rand(3)) for _ in range(10)]
             JSONFileSplit = os.path.splitext(fileRead)
             del resultList[:]
             del resultFrame[:]
             del summaryList[:]
             tic = time.clock()
             temppath=os.path.dirname(fileRead)+"\\"+fileExtension[0]+'.mp4'
             resultTempPath=os.path.join('D:\\UPC2\\RESULT' ,''+ fileExtension[0]+'.mp4')
             print(temppath)
             capture = cv2.VideoCapture(temppath)
             w = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
             h = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
             fourcc = cv2.VideoWriter_fourcc(*'H264')
             out = cv2.VideoWriter(resultTempPath,fourcc, capture.get(cv2.CAP_PROP_FPS), (int(w),int(h)))
            
             while True:
                 stime = time.time()
                 ret, frame = capture.read()
                 if frame is None:
                     break
                 results = tfnet.return_predict(frame)
                 
                 if ret:
                     draw_boxes(colors, results, frame,fileRead)
                     out.write(frame)
                    # cv2.imshow("predicted", frame)
                   
                    
                    # print('FPS {:.1f}'.format(1 / (time.time() - stime)))
                   
                 if cv2.waitKey(1) & 0xFF == ord('q'):
                     break
             toc = time. clock()
             print(toc - tic)
             capture.release()
             cv2.destroyAllWindows()

            # print("RESULT LIST")
             #print(summaryList)
             with open(fileRead) as temp_file:
                data = json.load(temp_file)
                data['results'] = {}
                data['summary']=summaryList
                data['results']['type'] = "image"
                data['results']['result'] = resultList
        
             with open(os.path.join('D:\\UPC2\\RESULT\\' + os.path.basename(JSONFileSplit[0]+".json")), 'w') as outfile:
                json.dump(data, outfile)
       
        


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
