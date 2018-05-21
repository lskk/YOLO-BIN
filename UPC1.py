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
    'model': 'cfg/yolo.cfg',
    'load': 'bin/yolo.weights',
    'threshold': 0.3,
    'gpu': 0.8,
    'gpuName': '/gpu:1'
}

summary=[]
def draw_boxes(colors, results, frame, filePath):
    # arduino = serial.Serial('COM3',9600)
    json_file = []
    JSONFileSplit = os.path.splitext(filePath)

    for (color, result) in zip(colors, results):
        # Convert confidence level to int from float
        json_temp = dict(result)
        x1 = json_temp['topleft']['x']
        x2 = json_temp["bottomright"]["x"]
        y1 = json_temp['topleft']['y']
        y2 = json_temp['bottomright']['y']
        yuhulabel=json_temp['label']
            # if yuhulabel in yuhuArrayLabel
            # else:
            #     yuhuArrayLabel.append(yuhulabel)
        print(summary)
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
        # till here
        json_file.append(json_temp)
        tl = (result['topleft']['x'], result['topleft']['y'])
       # tr = (result['topright']['x'],result['topright']['y'])
        br = (result['bottomright']['x'], result['bottomright']['y'])
        #result['topright']=(result['topleft']['x'], result['topleft']['y'])
       # bl = (result['bottomleft']['x'],result['bottomleft']['y'])
        #ctr = (result['center']['x'],result['center']['y'])
        #Xc = ((result['topleft']['x']-result['bottomright']['x'])/2 + result['topleft']['x'])
        #Yx = ((result['topleft']['y']-result['bottomright']['y'])/2 + result['topleft']['y'])
        #center = 
        confidence = result['confidence']
        label = result['label']
        # print('{},{},{},{}\n'.format(result['topleft']['x'], result['topleft']['y'],result['bottomright']['x'], result['bottomright']['y']))
        text = '{}: {:.1f}%'.format(label, confidence*100)
        frame = cv2.rectangle(frame, tl, br, color, 5)
        frame = cv2.putText(
            frame, text, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
        #print(result)
        #sasageyo = json.dumps(json_file)
        #print (sasageyo)
        # API
        # r = requests.post("http://167.205.7.227:8070/api/getFacebookData",json={'label' : json_file})
    #cv2.imshow("predicted", frame)
       # nameFile = spliteGo
    # with open('data.json', 'w') as outfile:
        #sasageyo = json.dump(json_file)
        #json.dump(json_file, outfile)
    # if (nameFile ==  fileExtension[1] == '.json' ):

    with open(JSONFileSplit[0]+".json") as temp_file:
        data = json.load(temp_file)
        data['results'] = {}
        tempJarraySum=[]
        tempJarraySum.append(summary)
        data['summary']=tempJarraySum
        tempJArray=[]
        tempJArray.append(json_file)
        data['results']['type'] = "image"
        data['results']['result'] = tempJArray
       # print('D:\\UPC1\\RESULT\\'+ os.path.basename(JSONFileSplit[0]+".json"))
       # wololo.update({'result':sasageyo})
    with open(os.path.join('D:\\UPC1\\RESULT\\' + os.path.basename(JSONFileSplit[0]+".json")), 'w') as outfile:
        json.dump(data, outfile)
        # print(JSONFileSplit[0]+".json")
        # print(JSONFileSplit[1])


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
        fileRead = (event.src_path)  # read src
        # print(fileRead)
        # splite source
        spliteGo = os.path.basename(fileRead)
            # print(spliteGo)
        fileExtension = os.path.splitext(spliteGo)
            # print(fileExtension)
            # print(event.src_path)
            # fileInput = (fileRead)
            # fileInput.split("\\")
            # go = fileInput.split("\\")
            # print(go)

        if (fileExtension[1] == '.jpg' or fileExtension[1] == '.png' or fileExtension[1] == '.jpeg'):
            del summary[:]
            tfnet = TFNet(options)

            colors = [tuple(255 * np.random.rand(3)) for _ in range(10)]
            print("samlekom mamang")
            frame = cv2.imread(fileRead)
            results = tfnet.return_predict(frame)
            print(results)
            draw_boxes(colors, results, frame, fileRead)

            print(spliteGo)
            #path = '\\result'
            #print (os.path.join(path,''+spliteGo))
            cv2.imwrite(os.path.join('D:\\UPC1\\RESULT', '' + spliteGo), frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


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
