import tkinter as tk
from tkinter import *
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

class App(Frame):
    def __init__(self,master=None):
        Frame.__init__(self, master)
        self.master = master
        self.label = Label(text="",width = 45, height = 6, bg = "white",font = ('Times', 25))
        self.label.place(x=100,y=310)
        self.label1 = Label(text="STEP 1",width = 18, height = 3, font = ('Times', 14))
        self.label1.place(x=100,y=170)
        self.label2 = Label(text="STEP 2",width = 18, height = 3, font = ('Times', 14))
        self.label2.place(x=550,y=170)
        self.label3 = Label(text="RESULT",width = 18, height = 3, font = ('Times', 14))
        self.label3.place(x=1000,y=170)
        self.label5 = Label(text="SAFETYFIRST TEMPERATURE MONITORING SYSTEM",width = 52, height = 3, bg = "white",font = ('Times', 27))
        self.label5.place(x=250,y=40)
        self.step1()
      
    def step1(self):
        
        self.label.configure(text="STEP1: Scan your Face or QR code")
        self.label1.configure(bg = "lightgreen")
        self.label2.configure(bg = "lightgrey")
        self.label3.configure(bg = "lightgrey")
        GPIO.output(4,GPIO.HIGH)
        GPIO.output(5,GPIO.LOW)
        GPIO.output(19,GPIO.LOW)
        GPIO.output(26,GPIO.LOW)
        self.after(800, self.cv)
        
    def cv(self):
            
        cam = cv2.VideoCapture(0)
        font = cv2.FONT_HERSHEY_SIMPLEX
        detector = cv2.QRCodeDetector()
                
        recognizer = cv2.face.LBPHFaceRecognizer_create()#cv2.createLBPHFaceRecognizer()
        recognizer.read("TrainingImageLabel/Trainner.yml")
        harcascadePath = "haarcascade_frontalface_default.xml"
        faceCascade = cv2.CascadeClassifier(harcascadePath);    
        df=pd.read_csv("StudentDetails/StudentDetails.csv")
              
        col_names =  ['Id','Name','Date','Time']
        attendance = pd.DataFrame(columns = col_names)
        
        while True:
            
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
                    cam.release()
                    cv2.destroyAllWindows()
                    self.after(800, self.step2)
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
                    self.after(800, self.step2)
                    print("end face")
                    break
                else:
                    Id='Unknown'                
                    tt=str(Id)  
                if(conf > 75):
                    noOfFile=len(os.listdir("ImagesUnknown"))+1
                    cv2.imwrite("ImagesUnknown/Image"+str(noOfFile) + ".jpg", im[y:y+h,x:x+w])            
                cv2.putText(im,str(tt),(x,y+h), font, 1,(255,255,255),2)        
            attendance=attendance.drop_duplicates(subset=['Id'],keep='first')    
            winname = "im"
            cv2.namedWindow(winname)        # Create a named window
            cv2.moveWindow(winname, 920,330)  # Move it to (40,30)
            im = cv2.resize(im, (350,300))
            cv2.imshow('im',im)
            if (cv2.waitKey(1)==ord('q')):
                break

    def step2(self):
        cam = cv2.VideoCapture(0)
        font = cv2.FONT_HERSHEY_SIMPLEX
        detector = cv2.QRCodeDetector()
        cam.release()
        cv2.destroyAllWindows()
        print("11")
        self.label1.configure(bg = "lightgreen")
        self.label2.configure(bg = "lightgreen")
        self.label3.configure(bg = "lightgrey")
        GPIO.output(4,GPIO.HIGH)
        GPIO.output(5,GPIO.HIGH)
        GPIO.output(19,GPIO.LOW)
        GPIO.output(26,GPIO.LOW)
        self.my_pic = Image.open("/home/pi/Face-Recognition-Based-Attendance-System/temp.jpg")
        self.resized = self.my_pic.resize((300,300),Image.ANTIALIAS)
        self.new_pic = ImageTk.PhotoImage(self.resized)
        self.my_label = Label(root, image = self.new_pic, bg = "white")
        self.my_label.place(x=920,y=280)
        self.after(100, self.dist)
        
    def dist(self):
        dist = distance()
        self.label.configure(text="STEP 2: Please bring your head closer to the system.\nNote: Distance should be between 1 to 5 cms\nDistance = %.1f cm" % dist)
            
        if dist<5: 
            print("dist %.1f"%dist)
            self.after(50, self.step3)
        else:
            print("dist %.1f"%dist)
            print("else")
            self.after(500, self.dist)
            
    def step3(self):
     
        sensor = MLX90614()
        print(sensor.get_obj_temp())
        time.sleep(0.2)
        val=usrdata.data
        
        if sensor.get_obj_temp() < 32:
            self.label.configure(text="You are good to go. Thanks for coopreration\nBody Temperature: %.1f" %sensor.get_obj_temp())
            self.label1.configure(bg = "lightgreen")
            self.label2.configure(bg = "lightgreen")
            self.label3.configure(bg = "lightgreen")
            self.my_pic = Image.open("/home/pi/Face-Recognition-Based-Attendance-System/safe.jpg")
            self.resized = self.my_pic.resize((300,300),Image.ANTIALIAS)
            self.new_pic = ImageTk.PhotoImage(self.resized)
            self.my_label = Label(root, image = self.new_pic, bg = "white")
            self.my_label.place(x=920,y=280)
            print("safe")
            print(sensor.get_obj_temp())
           
            GPIO.output(4,GPIO.HIGH)
            GPIO.output(5,GPIO.HIGH)
            GPIO.output(13,GPIO.HIGH)
            GPIO.output(26,GPIO.LOW)
            GPIO.output(19,GPIO.HIGH)
            time.sleep(0.3)
            GPIO.output(19,GPIO.LOW)            
            result = firebase.put("/users/{}".format(val),"temp",sensor.get_obj_temp())
            self.after(500, self.stopp)
            
        else:
            self.label.configure(text="Fever Detected! Please stay home\nBody Temperature: %.1f" %sensor.get_obj_temp())
            self.label1.configure(bg = "lightgreen")
            self.label2.configure(bg = "lightgreen")
            self.label3.configure(bg = "red")
            self.my_pic = Image.open("/home/pi/Face-Recognition-Based-Attendance-System/fever.jpg")
            self.resized = self.my_pic.resize((300,300),Image.ANTIALIAS)
            self.new_pic = ImageTk.PhotoImage(self.resized)
            self.my_label = Label(root, image = self.new_pic, bg = "white")
            self.my_label.place(x=920,y=280)

            print("Fever")
            print(sensor.get_obj_temp())
            GPIO.output(4,GPIO.HIGH)
            GPIO.output(5,GPIO.HIGH)
            GPIO.output(13,GPIO.LOW)
            GPIO.output(26,GPIO.HIGH)
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
            self.after(500, self.stopp)
            
    def stopp(self):
        
        self.after(3000, self.step1)
        time.sleep(3)
root = Tk()
root.configure(background='white')
app=App(root)
root.wm_title("Temperature Monitoring System")
root.geometry("1370x710+0+0")

my_pic = Image.open("/home/pi/Face-Recognition-Based-Attendance-System/logo.png")
resized = my_pic.resize((150,150),Image.ANTIALIAS)
new_pic = ImageTk.PhotoImage(resized)
my_label = Label(root, image = new_pic, bg = "white")
my_label.place(x=65,y=15)


 
root.mainloop()
