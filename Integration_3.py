import os
import cv2
from darkflow.net.build import TFNet
from finding_lane_2 import detect_lanes_img
import numpy as np
import serial
import time
import json

    

def draw_boxes(colors,results,frame,xm_per_pix):
    #arduino = serial.Serial('COM3',9600)
    json_file = []
    frame_width = frame.shape[1]
    print(frame_width)
    for (color, result) in zip (colors, results):
        #Convert confidence level to int from float 
        json_temp = dict(result)
        json_conf = json_temp['confidence']
        json_conf = int(round(json_conf*100))
        json_temp['confidence'] = json_conf
        #till here
        json_file.append(json_temp)
        tl = (result['topleft']['x'], result['topleft']['y'])
        br = (result['bottomright']['x'], result['bottomright']['y'])
        x_pix = result['bottomright']['y']
        x_pix = abs(frame_width - x_pix)
        x = x_pix * xm_per_pix
        label = result['label']
        #confidence = result['confidence']
        text = '{}: {:.0f}m'.format(label, x)
        frame = cv2.rectangle(frame, tl, br, color, 5)
        frame = cv2.putText(frame, text, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
        '''
        if (arduino.isOpen()):
            word = label + '\n'
            bword = bytes(word,'utf-8')
            arduino.write(bword)
        '''
    #cv2.imshow ("predicted",frame)
    with open('data.json','w') as outfile:
        json.dump(json_file,outfile)
    #arduino.close()
'''
def send_to_arduino(word):
    arduino = serial.Serial('COM3',9600)
    if (arduino.isOpen()):
        word = word +'\n'
        bword = bytes(word,'utf-8')
        arduino.write(bword)

def result_to_word(result):
    word = '{}'.format(result['label'])
    return word
'''


# M A I N   P R O G R A M
options = {
    'model': 'cfg/tiny-yolo-voc.cfg',
    'load': 'bin/tiny-yolo-voc.weights',
    'threshold': 0.3,
    'gpu': 0.5
}

tfnet = TFNet(options)
fileRead = input('Enter Image/Video/Webcam : ')
filename , fileExtension = os.path.splitext(fileRead)

colors = [tuple(255 * np.random.rand(3)) for _ in range(10)]

if (fileExtension == '.jpg') :
    frame = cv2.imread (fileRead)
    results = tfnet.return_predict(frame)
    draw_boxes(colors,results,frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
elif (fileExtension == '.mp4') :
    capture = cv2.VideoCapture(fileRead)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    while True:
        stime = time.time()
        ret, frame = capture.read()
        results = tfnet.return_predict(frame)
        if ret:
            frame, xm_per_pix = detect_lanes_img(frame)
            draw_boxes(colors,results,frame,xm_per_pix)
            cv2.imshow ("predicted",frame)
            print('FPS {:.1f}'.format(1 / (time.time() - stime)))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    capture.release()
    cv2.destroyAllWindows()
elif (fileExtension == '.0' or fileExtension == '.1' ) :
    fileExtension = fileExtension[-1]
    cam = int (fileExtension)
    capture = cv2.VideoCapture(cam)
    while True:
        stime = time.time()
        ret, frame = capture.read()
        results = tfnet.return_predict(frame)
        if ret:
            frame, xm_per_pix = detect_lanes_img(frame)
            draw_boxes(colors,results,frame,xm_per_pix)
            cv2.imshow ("predicted",frame)
            print('FPS {:.1f}'.format(1 / (time.time() - stime)))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    capture.release()
    cv2.destroyAllWindows()
else:
    print('Input Invalid')



