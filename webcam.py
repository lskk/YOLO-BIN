import os
import cv2

fileRead = input()
filename , fileExtension = os.path.splitext(fileRead)

if (fileExtension == '.jpg') :
    img = cv2.imread (fileRead)
    cv2.imshow ("image",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
elif (fileExtension == '.mp4') :
    capture = cv2.VideoCapture(fileRead)
    while True:
        ret, frame = capture.read()
        if ret:
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    capture.release()
    cv2.destroyAllWindows()
elif (fileExtension == '.0') :
    fileExtension = fileExtension[-1]
    cam = int (fileExtension)
    capture = cv2.VideoCapture(cam)
    while True:
        ret, frame = capture.read()
        if ret:
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    capture.release()
    cv2.destroyAllWindows()
else:
    print('Input Invalid')



