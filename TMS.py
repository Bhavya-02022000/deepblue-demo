import tkinter as tk
from tkinter import Message ,Text
import cv2,os
import shutil
import csv
import numpy as np
from PIL import Image, ImageTk
import pandas as pd
import datetime
import time
import tkinter.ttk as ttk
import tkinter.font as font

import RPi.GPIO as GPIO
import smbus

from firebase import firebase
import numpy as np
from PIL import Image, ImageTk
import pandas as pd

url = 'https://deepblue-2621f-default-rtdb.firebaseio.com/'
firebase = firebase.FirebaseApplication(url)
result = firebase.put("/Test Val","Value",55)
print(result)

GPIO.setmode(GPIO.BCM)

GPIO.setwarnings(False)
GPIO.setup(4,GPIO.OUT)
GPIO.setup(5,GPIO.OUT)
GPIO.setup(13,GPIO.OUT)
GPIO.setup(26,GPIO.OUT)
GPIO.setup(19,GPIO.OUT)

GPIO.output(19,GPIO.LOW)
GPIO.output(4,GPIO.LOW)
GPIO.output(5,GPIO.LOW)
GPIO.output(13,GPIO.LOW)
GPIO.output(26,GPIO.LOW)


#set GPIO Pins
GPIO_TRIGGER = 20
GPIO_ECHO = 21
 
#set GPIO direction (IN / OUT)
GPIO.setup(GPIO_TRIGGER, GPIO.OUT)
GPIO.setup(GPIO_ECHO, GPIO.IN)

class usrdata():
    def data(data):
        data
        return data

def distance():
    # set Trigger to HIGH
    GPIO.output(GPIO_TRIGGER, True)
 
    # set Trigger after 0.01ms to LOW
    time.sleep(0.00001)
    GPIO.output(GPIO_TRIGGER, False)
 
    StartTime = time.time()
    StopTime = time.time()
 
    # save StartTime
    while GPIO.input(GPIO_ECHO) == 0:
        StartTime = time.time()
 
    # save time of arrival
    while GPIO.input(GPIO_ECHO) == 1:
        StopTime = time.time()
 
    # time difference between start and arrival
    TimeElapsed = StopTime - StartTime
    # multiply with the sonic speed (34300 cm/s)
    # and divide by 2, because there and back
    distance = (TimeElapsed * 34300) / 2
 
    return distance


 
class MLX90614():
 
    MLX90614_RAWIR1=0x04
    MLX90614_RAWIR2=0x05
    MLX90614_TA=0x06
    MLX90614_TOBJ1=0x07
    MLX90614_TOBJ2=0x08
 
    MLX90614_TOMAX=0x20
    MLX90614_TOMIN=0x21
    MLX90614_PWMCTRL=0x22
    MLX90614_TARANGE=0x23
    MLX90614_EMISS=0x24
    MLX90614_CONFIG=0x25
    MLX90614_ADDR=0x0E
    MLX90614_ID1=0x3C
    MLX90614_ID2=0x3D
    MLX90614_ID3=0x3E
    MLX90614_ID4=0x3F
 
    def __init__(self, address=0x5a, bus_num=1):
        self.bus_num = bus_num
        self.address = address
        self.bus = smbus.SMBus(bus=bus_num)
 
    def read_reg(self, reg_addr):
        return self.bus.read_word_data(self.address, reg_addr)
 
    def data_to_temp(self, data):
        temp = (data*0.02) - 273.15
        return temp
 
    def get_amb_temp(self):
        data = self.read_reg(self.MLX90614_TA)
        return self.data_to_temp(data)
 
    def get_obj_temp(self):
        data = self.read_reg(self.MLX90614_TOBJ1)
        return self.data_to_temp(data)

 

recognizer = cv2.face.LBPHFaceRecognizer_create()#cv2.createLBPHFaceRecognizer()
recognizer.read("TrainingImageLabel/Trainner.yml")
harcascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(harcascadePath);    
df=pd.read_csv("StudentDetails/StudentDetails.csv")
      
col_names =  ['Id','Name','Date','Time']
attendance = pd.DataFrame(columns = col_names)


temp=0
count=0
cvv=0
temp_print=0
temp_count=0

while True:
    cam = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    detector = cv2.QRCodeDetector()
    while temp==0:
        
        GPIO.output(4,GPIO.LOW)
        GPIO.output(5,GPIO.LOW)
        GPIO.output(13,GPIO.LOW)
        GPIO.output(26,GPIO.LOW)
         
        ret, im =cam.read()
        gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        faces=faceCascade.detectMultiScale(gray, 1.2,5)
        data, bbox, _ = detector.detectAndDecode(im)
        if(bbox is not None):
            for i in range(len(bbox)):
                cv2.line(im, tuple(bbox[i][0]), tuple(bbox[(i+1) % len(bbox)][0]), color=(255,0, 255), thickness=2)
            cv2.putText(im, data, (int(bbox[0][0][0]), int(bbox[0][0][1]) - 10), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 255, 0), 2)
        if data:
                print("data found: ", data)
                usrdata.data=data
                GPIO.output(19,GPIO.HIGH)
                time.sleep(0.5)
                GPIO.output(4,GPIO.HIGH)
                GPIO.output(5,GPIO.LOW)
                GPIO.output(13,GPIO.LOW)
                GPIO.output(26,GPIO.LOW)
                GPIO.output(19,GPIO.LOW)
                
                temp=1
                cam.release()
                cv2.destroyAllWindows()
                break
        for(x,y,w,h) in faces:
            cv2.rectangle(im,(x,y),(x+w,y+h),(225,0,0),2)
            Id, conf = recognizer.predict(gray[y:y+h,x:x+w])                                   
            if(conf < 50):
                ts = time.time()      
                date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                aa=df.loc[df['Id'] == Id]['Name'].values
                tt=str(Id)+"-"+aa
                attendance.loc[len(attendance)] = [Id,aa,date,timeStamp]
                temp=1
                print(aa)
                print(aa[0])
                usrdata.data=aa[0]
                
                GPIO.output(19,GPIO.HIGH)
                time.sleep(0.5)
                GPIO.output(4,GPIO.HIGH)
                GPIO.output(5,GPIO.LOW)
                GPIO.output(13,GPIO.LOW)
                GPIO.output(26,GPIO.LOW)
                GPIO.output(19,GPIO.LOW)
                cam.release()
                cv2.destroyAllWindows()
                
            else:
                Id='Unknown'                
                tt=str(Id)  
            if(conf > 75):
                noOfFile=len(os.listdir("ImagesUnknown"))+1
                cv2.imwrite("ImagesUnknown/Image"+str(noOfFile) + ".jpg", im[y:y+h,x:x+w])            
            cv2.putText(im,str(tt),(x,y+h), font, 1,(255,255,255),2)        
        attendance=attendance.drop_duplicates(subset=['Id'],keep='first')    
        cv2.imshow('im',im) 
        if (cv2.waitKey(1)==ord('q')):
            break

    while temp==1:
        dist = distance()
        dist = round(dist,1)
        time.sleep(0.5)
        print ("Measured Distance = %.1f cm" % dist)
      
        if dist<30 and dist>20:
            count=count+1
            
        else:
            count=0
            
        if count==5:
            temp=2
            GPIO.output(4,GPIO.HIGH)
            GPIO.output(5,GPIO.HIGH)
            GPIO.output(13,GPIO.LOW)
            GPIO.output(26,GPIO.LOW)
            GPIO.output(19,GPIO.HIGH)
            time.sleep(0.5)
            GPIO.output(19,GPIO.LOW)
            print("2")
        #ultra
    while temp==2:
        
        sensor = MLX90614()
        print(sensor.get_obj_temp())
        time.sleep(0.2)
        temp_print=temp_print+1
        val=usrdata.data
        if sensor.get_obj_temp() < 30:
            
            print("safe")
            print(sensor.get_obj_temp())
            temp=0
            GPIO.output(4,GPIO.HIGH)
            GPIO.output(5,GPIO.HIGH)
            GPIO.output(13,GPIO.HIGH)
            GPIO.output(26,GPIO.LOW)
            GPIO.output(19,GPIO.HIGH)
            time.sleep(0.3)
            GPIO.output(19,GPIO.LOW)
            result = firebase.put("/users/{}".format(val),"temp",sensor.get_obj_temp())
            time.sleep(5)
        else:
            print("Fever")
            print(sensor.get_obj_temp())
            temp=0
            GPIO.output(4,GPIO.HIGH)
            GPIO.output(5,GPIO.HIGH)
            GPIO.output(13,GPIO.HIGH)
            GPIO.output(26,GPIO.LOW)
            result = firebase.put("/users/{}".format(val),"temp",sensor.get_obj_temp())
            GPIO.output(19,GPIO.HIGH)
            time.sleep(0.3)
            GPIO.output(19,GPIO.LOW)
            time.sleep(0.3)
            GPIO.output(19,GPIO.HIGH)
            time.sleep(0.3)
            GPIO.output(19,GPIO.LOW)
            time.sleep(0.3)
            GPIO.output(19,GPIO.HIGH)
            time.sleep(0.3)
            GPIO.output(19,GPIO.LOW)

        
        
