import os
import cv2
from darkflow.net.build import TFNet
#from finding_lane_1 import detect_lanes_img
import numpy as np
#import serial
import time
import json

    

def draw_boxes_image(colors,results,frame):
    #arduino = serial.Serial('COM3',9600)
    json_file = []
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
        confidence = result['confidence']
        label = result['label']
        text = '{}: {:.1f}%'.format(label,confidence*100 )
        frame = cv2.rectangle(frame, tl, br, color, 5)
        frame = cv2.putText(frame, text, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
        '''
        if (arduino.isOpen()):
            word = label + '\n'
            bword = bytes(word,'utf-8')
            arduino.write(bword)
        '''
    cv2.imshow ("predicted",frame)
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
     'model': 'cfg/yolo-voc.cfg',
    'load': 'bin/yolo-voc.weights',
    'threshold': 0.3,
    'gpu': 0.5
}

tfnet = TFNet(options)
fileRead = input('Enter Image/Video/Webcam : ')
filename , fileExtension = os.path.splitext(fileRead)

colors = [tuple(255 * np.random.rand(3)) for _ in range(10)]

if (fileExtension == '.jpg') :
    frame = cv2.imread (fileRead)
   # frame = detect_lanes_img(frame)
    results = tfnet.return_predict(frame)
    draw_boxes_image(colors,results,frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
elif (fileExtension == '.mp4') :
    capture = cv2.VideoCapture(fileRead)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    frame_number = 1
    json_file = {}
    px_cm_person = 6047
    while True:
        stime = time.time()
        ret, frame = capture.read()
        results = tfnet.return_predict(frame)
        if ret:
            #
            #frame = detect_lanes_img(frame)
            json_frame = []
            frame_str = 'frame {}'.format(frame_number)
            for (color, result) in zip (colors, results):
                #Convert confidence level to int from float 
                json_temp = dict(result)
                json_conf = json_temp['confidence']
                json_conf = int(round(json_conf*100))
                json_temp['confidence'] = json_conf
                json_frame.append(json_temp)
                tl = (result['topleft']['x'], result['topleft']['y'])
                br = (result['bottomright']['x'], result['bottomright']['y'])
                confidence = result['confidence']
                label = result['label']
                coorX= result['topleft']['x'] + result['bottomright']['x']/2
                coorY= result['topleft']['y'] + result['bottomright']['y']/2
                centroids = (coorX,coorY)
                if (label == 'person'): 
                    width_person = result['bottomright']['y'] - result['topleft']['y']
                    dist_person = px_cm_person/width_person
                    text = '{}: {:.1f}%'.format(label,dist_person )
                else :
                    text = '{}: {:.1f}%'.format(label,confidence*100 )
                
                frame = cv2.rectangle(frame, tl, br, color, 5)
                frame = cv2.putText(frame, text, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
                '''
                if (arduino.isOpen()):
                    word = label + '\n'
                    bword = bytes(word,'utf-8')
                    arduino.write(bword)
                    '''
            json_file[frame_str] = json_frame 
            frame_number = frame_number + 1
            cv2.imshow ("predicted",frame)
            print('FPS {:.1f}'.format(1 / (time.time() - stime)))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    capture.release()
    cv2.destroyAllWindows()
    with open('data.json','w') as outfile:
        json.dump(json_file,outfile)
    print('Jumlah Frame :{}'.format(frame_number))
    #arduino.close()
elif (fileExtension == '.0' or fileExtension == '.1' ) :
    fileExtension = fileExtension[-1]
    cam = int (fileExtension)
    capture = cv2.VideoCapture(cam)
    frame_number = 1
    json_file = {}
    px_cm_person = 20000
    while True:
        stime = time.time()
        ret, frame = capture.read()
        results = tfnet.return_predict(frame)
        if ret:
            #frame = detect_lanes_img(frame)
            json_frame = []
            frame_str = 'frame {}'.format(frame_number)
            for (color, result) in zip (colors, results):
                #Convert confidence level to int from float 
                json_temp = dict(result)
                json_conf = json_temp['confidence']
                json_conf = int(round(json_conf*100))
                json_temp['confidence'] = json_conf
                json_frame.append(json_temp)
                tl = (result['topleft']['x'], result['topleft']['y'])
                br = (result['bottomright']['x'], result['bottomright']['y'])
                confidence = result['confidence']
                label = result['label']
                coorX= result['topleft']['x'] + result['bottomright']['x']/2
                coorY= result['topleft']['y'] + result['bottomright']['y']/2
                centroids = (coorX,coorY)
                frame = cv2.rectangle(frame, tl, br, color, 5)
                if (label == 'person'): 
                    width_person = result['bottomright']['y'] - result['topleft']['y']
                    dist_person = px_cm_person/width_person
                    dist_person = format(dist_person, '.2f')
                    print(dist_person)
                    text2 = '{}: {}cm'.format(label, dist_person )
                    print(text2)
                    frame = cv2.putText(frame, text2, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
                else :
                    text = '{}: {:.1f}%'.format(label,confidence*100 )
                    frame = cv2.putText(frame, text, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
                
                '''
                if (arduino.isOpen()):
                    word = label + '\n'
                    bword = bytes(word,'utf-8')
                    arduino.write(bword)
                    '''
            json_file[frame_str] = json_frame 
            frame_number = frame_number + 1
            cv2.imshow ("predicted",frame)
            print('FPS {:.1f}'.format(1 / (time.time() - stime)))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    capture.release()
    cv2.destroyAllWindows()
    with open('data.json','w') as outfile:
        json.dump(json_file,outfile)
    print('Jumlah Frame :{}'.format(frame_number))
    #arduino.close()
elif (fileRead == 'exit'):
    pass
else:
    print('Input Invalid')



