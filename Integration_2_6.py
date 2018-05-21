import os
import cv2
from darkflow.net.build import TFNet
from finding_lane_1 import detect_lanes_img
import numpy as np
import serial
import time
import json
import pika
import threading
from threading import Timer,Thread,Event

#inisiasi global variable
json_rmq = {}

class perpetualTimer(): #Fungsi untuk melakukan threading publishin data json deteksi YOLO

   def __init__(self,t,hFunction):
      self.t=t
      self.hFunction = hFunction
      self.thread = Timer(self.t,self.handle_function)

   def handle_function(self):
      self.hFunction()
      self.thread = Timer(self.t,self.handle_function)
      self.thread.start()

   def start(self):
      self.thread.start()

   def cancel(self):
      self.thread.cancel()

def publish_json(): #Prosedur untuk melakukan publishing
    global json_rmq
    global channel
    ''' global arduino '''
    if(bool(json_rmq)):
        channel.basic_publish(exchange='amq.topic',routing_key='data.json',body=json.dumps(json_rmq))
        ''' words = json.dumps(json_rmq)
        bword = bytes(words,'utf-8')
        arduino.write(bword) '''
        json_rmq = {}

def draw_boxes_image(colors,results,frame,channel):
    #arduino = serial.Serial('COM3',9600)
    json_file = []
    #properties = pika.BasicProperties(content_type = "application/json",delivery_mode = 1)
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
    channel.basic_publish(exchange='amq.topic',routing_key='data.json',body=json.dumps(json_file))
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

tfnet = TFNet(options) #inisiasi darkflow
fileRead = input('Enter Image/Video/Webcam : ') #membaca nama file dari pengguna
filename , fileExtension = os.path.splitext(fileRead) #memisahkan nama file dan extension
px_cm_car = 15118 #cm per pixel for car
colors = [tuple(255 * np.random.rand(3)) for _ in range(10)] 

#Inisiasi  rabbitmq dan arduino
credentials = pika.PlainCredentials('autodrive', 'autodrive2218!')
parameters = pika.ConnectionParameters('167.205.7.226',5672,'/autodrive',credentials)

connection = pika.BlockingConnection(parameters)
channel = connection.channel()

channel.queue_declare(queue='data.YOLO', durable=True)

channel.exchange_declare(exchange='amq.topic', exchange_type='topic', durable=True)

arduino = serial.Serial('COM15',9600)
time.sleep(1) 

if (fileExtension == '.jpg') :
    frame = cv2.imread (fileRead)
    results = tfnet.return_predict(frame)
    draw_boxes_image(colors,results,frame,channel)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
elif (fileExtension == '.mp4') :
    capture = cv2.VideoCapture(fileRead)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    frame_number = 1
    json_file = {}
    px_cm_person = 7500
    ztime = time.time()
    pub_json = perpetualTimer(1,publish_json)
    pub_json.start()
    while True:
        stime = time.time()
        ret, frame = capture.read()
        if ret:
            cols = frame.shape[1]
            cols_decision1 = cols/2 + 20
            cols_decision2 = cols/2 - 20
            results = tfnet.return_predict(frame)
            frame = detect_lanes_img(frame)
            json_frame = []
            frame_str = 'frame {}'.format(frame_number)
            word_frame = ''
            for (color, result) in zip (colors, results):
                #Convert confidence level to int from float 
                json_temp = dict(result)
                json_conf = json_temp['confidence']
                json_conf = int(round(json_conf*100))
                json_temp['confidence'] = json_conf
                json_frame.append(json_temp)
                tl = (result['topleft']['x'], result['topleft']['y'])
                br = (result['bottomright']['x'], result['bottomright']['y'])
                cols_center = (tl[0] + br[0])/2
                if (cols_center > cols_decision2 and cols_center < cols_decision1):
                    obj_position = 0
                elif (cols_center >= cols_decision1):
                    obj_position = 1
                elif (cols_center <= cols_decision2):
                    obj_position = -1
                confidence = result['confidence']
                label = result['label']
                if (label == 'person'): 
                    width_person = result['bottomright']['y'] - result['topleft']['y']
                    dist_person = px_cm_person/width_person
                    dist_person = format(dist_person, '.2f')
                    text = '{}: {}cm'.format(label,dist_person )
                    word = label + ',' + '{}'.format(confidence) +','+ dist_person + ',' +'{}'.format(obj_position)
                elif (label == 'car'):
                    width_car = result['bottomright']['y'] - result['topleft']['y']
                    dist_car = format(px_cm_car/width_car, '.2f')
                    text = '{}: {}cm'.format(label,dist_car)
                    word = label + ',' + '{}'.format(confidence) +','+ dist_car  + ',' + '{}'.format(obj_position)
                else :
                    text = '{}: {:.1f}%'.format(label,confidence*100 )
                    word = label + ',' + '{}'.format(confidence) + ','+ 'not a car or person'  + ',' +  '{}'.format(obj_position)
                
                frame = cv2.rectangle(frame, tl, br, color, 5)
                frame = cv2.putText(frame, text, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
                
                word_frame = word_frame + word + ';'
                '''
                if (arduino.isOpen()):
                    word = label + '\n'
                    bword = bytes(word,'utf-8')
                    arduino.write(bword)
                    '''
            json_file[frame_str] = list(json_frame)
            json_rmq[frame_str] = list(json_frame) 

            word_frame = word_frame +  '\n'
            if (arduino.isOpen()):
                    bword = bytes(word_frame,'utf-8')
                    arduino.write(bword)

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
    avg_frame = format( frame_number/(time.time()-ztime) ,'.2f')
    print('Average FPS: {}'.format(avg_frame))
    pub_json.cancel()
    arduino.close()
elif (fileExtension == '.0' or fileExtension == '.1' ) :
    fileExtension = fileExtension[-1]
    cam = int (fileExtension)
    capture = cv2.VideoCapture(cam)
    frame_number = 1
    json_file = {}
    #json_rmq = {}
    px_cm_person = 20000
    ztime = time.time()
    pub_json = perpetualTimer(1,publish_json)
    pub_json.start()
    while True:
        stime = time.time()
        ret, frame = capture.read()
        results = tfnet.return_predict(frame)
        if ret:
            cols = frame.shape[1]
            cols_decision1 = cols/2 + 20
            cols_decision2 = cols/2 - 20
            #frame = detect_lanes_img(frame)
            json_frame = []
            frame_str = 'frame {}'.format(frame_number)
            word_frame = ''
            for (color, result) in zip (colors, results):
                #Convert confidence level to int from float 
                json_temp = dict(result)
                json_conf = json_temp['confidence']
                json_conf = int(round(json_conf*100))
                json_temp['confidence'] = json_conf
                json_frame.append(json_temp)
                tl = (result['topleft']['x'], result['topleft']['y'])
                br = (result['bottomright']['x'], result['bottomright']['y'])
                cols_center = (tl[0] + br[0])/2
                if (cols_center > cols_decision2 and cols_center < cols_decision1):
                    obj_position = 0
                elif (cols_center >= cols_decision1):
                    obj_position = 1
                elif (cols_center <= cols_decision2):
                    obj_position = -1
                confidence = result['confidence']
                label = result['label']
                if (label == 'person'): 
                    width_person = result['bottomright']['y'] - result['topleft']['y']
                    dist_person = px_cm_person/width_person
                    dist_person = format(dist_person, '.2f')
                    text = '{}: {}cm'.format(label,dist_person )
                    word = label + ',' + '{}'.format(confidence) +','+ dist_person + ',' +'{}'.format(obj_position)
                elif (label == 'car'):
                    width_car = result['bottomright']['y'] - result['topleft']['y']
                    dist_car = format(px_cm_car/width_car, '.2f')
                    text = '{}: {}cm'.format(label,dist_car)
                    word = label + ',' + '{}'.format(confidence) +','+ dist_car  + ',' + '{}'.format(obj_position)
                else :
                    text = '{}: {:.1f}%'.format(label,confidence*100 )
                    word = label + ',' + '{}'.format(confidence) + ','+ 'not a car or person'  + ',' +  '{}'.format(obj_position)
                
                frame = cv2.rectangle(frame, tl, br, color, 5)
                frame = cv2.putText(frame, text, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
                
                word_frame = word_frame + word + ';'
                   
            json_file[frame_str] = list(json_frame)
            json_rmq[frame_str] = list(json_frame) 

            word_frame = word_frame +  '\n'
            if (arduino.isOpen()):
                    bword = bytes(word_frame,'utf-8')
                    arduino.write(bword) 
            # if ((frame_number % 5) == 0):
            #     channel.basic_publish(exchange='amq.topic',routing_key='data.json',body=json.dumps(json_rmq))
            #     json_rmq = {}
            frame_number = frame_number + 1
            cv2.imshow ("predicted",frame)
            print('FPS {:.1f}\n'.format(1 / (time.time() - stime)))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    capture.release()
    cv2.destroyAllWindows()
    with open('data.json','w') as outfile:
        json.dump(json_file,outfile)
    print('Jumlah Frame :{}'.format(frame_number))
    avg_frame = format( frame_number/(time.time()-ztime) ,'.2f')
    print('Average FPS: {}'.format(avg_frame))
    pub_json.cancel()
    arduino.close()
elif (fileRead == 'exit'):
    pass
else:
    print('Input Invalid')



