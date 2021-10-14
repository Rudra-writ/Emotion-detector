# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 02:36:16 2021

@author: 91629
"""


from keras.models import load_model
import cv2
import numpy as np
model = load_model('emotion detector')

face_clsfr=cv2.CascadeClassifier('your path to ---> haarcascade_frontalface_default.xml')

source=cv2.VideoCapture(0)
classes=['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

while(True):

    ret,img=source.read()
  
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face_clsfr.detectMultiScale(gray,1.3,5)  

    for (x,y,w,h) in faces:
    
        face=gray[y:y+h,x:x+w]
        resized=cv2.resize(face,(55,55))
        normalized=resized/255.0
        reshaped=np.reshape(normalized,(1,55,55,1))
        result=model.predict(reshaped)

        label=classes[np.argmax(result,axis=1)[0]]
      
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.rectangle(img,(x,y-40),(x+w,y),(0,255,0),-1)
        cv2.putText(img, label, (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
        
        
    cv2.imshow('Emotions',img)
    key=cv2.waitKey(1)
    
    if(key==27):
        break
        
cv2.destroyAllWindows()
source.release()