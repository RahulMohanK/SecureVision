import numpy as np 
import cv2
import matplotlib.pyplot as plt 
import time
import sys
import dlib
from imutils.video import VideoStream
from imutils import face_utils
import imutils
from mtcnn.mtcnn import MTCNN
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

def bb_to_rect(bb):
    top=bb[1]
    left=bb[0]
    right=bb[0]+bb[2]
    bottom=bb[1]+bb[3]

    a=np.array([(top, right),(bottom, left)]) 
    return ('[%s]' % ' '.join(map(str, a)))

def convertToRGB(img):
    return cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
def image_detector(img):
    haar_face_cascade_profile= cv2.CascadeClassifier('haarcascade_profileface.xml')
    haar_face_cascade_front= cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

    faces_p = haar_face_cascade_profile.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)
    faces_p = haar_face_cascade_front.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)   
    return faces_p

predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
detector = MTCNN()
detect = dlib.get_frontal_face_detector()

vs = cv2.VideoCapture(0)
time.sleep(2.0)

while(True):

    ret,frame = vs.read()
    frame = imutils.resize(frame,width=600)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # faces_p  = detector.detect_faces(frame)
    # faces_p = image_detector(gray)
    faces_p = detect(gray,0)
    for face in faces_p:
        
        (bX,bY,bW,bH) = face_utils.rect_to_bb(face)
        cv2.rectangle(frame,(bX,bY),(bX+bW,bY+bH),(0,255,0),1)

        shape = predictor(gray,face)
        shape=face_utils.shape_to_np(shape)

        for (i,(x,y)) in enumerate(shape):
            cv2.circle(frame,(x,y),1,(0,0,255),-1)
            cv2.putText(frame, str(i + 1), (x - 10, y - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)




    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break
	


cv2.destroyAllWindows()
vs.stop()
