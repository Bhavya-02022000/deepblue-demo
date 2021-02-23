from tkinter import *
import cv2,os,time
import shutil
import csv
import numpy as np
from PIL import Image, ImageTk
import pandas as pd
import datetime
import tkinter.ttk as ttk
import tkinter.font as font
from PIL import Image, ImageTk

class tempp():
    temp=''

class usrdata():
    def data(data):
        data
        return data


class App(Frame):
    def __init__(self,master=None):
        Frame.__init__(self, master)
        self.master = master
        self.label = Label(text="",width = 45, height = 6, bg = "white",font = ('Times', 25))
        self.label.place(x=270,y=460)       
        self.label5 = Label(text="SAFETYFIRST TEMPERATURE MONITORING SYSTEM",width = 52, height = 3, bg = "white",font = ('Times', 27))
        self.label5.place(x=230,y=30)
        print("init")
        self.step1()
    def step1(self):
        
        self.label.configure(text="Scan your QR code from your SafetyFirst App")
        self.my_pic = Image.open("/home/pi/Face-Recognition-Based-Attendance-System/qrcode.jpg")
        self.resized = self.my_pic.resize((300,300),Image.ANTIALIAS)
        self.new_pic = ImageTk.PhotoImage(self.resized)
        self.my_label = Label(root, image = self.new_pic, bg = "white")
        self.my_label.place(x=530,y=180)
        self.after(800, self.cv)
        
    def cv(self):
        print("cv")    
        cap = cv2.VideoCapture(0)
        detector = cv2.QRCodeDetector()
   
        while True:
            _, im = cap.read()
            data, bbox, _ = detector.detectAndDecode(im)
            
            if(bbox is not None):
                for i in range(len(bbox)):
                    cv2.line(im, tuple(bbox[i][0]), tuple(bbox[(i+1) % len(bbox)][0]), color=(255,
                             0, 255), thickness=2)
                cv2.putText(im, data, (int(bbox[0][0][0]), int(bbox[0][0][1]) - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 0), 2)
                if data:
                    print("data")
                    print("data found: ", data)
                    usrdata.data=data
                    cap.release()
                    cv2.destroyAllWindows()
                    self.after(500, self.step2())

            
        cap.release()
        cv2.destroyAllWindows()
        self.after(100, self.step1())


    def step2(self):
        
        self.label.configure(text="Please Bring your Face in front of Camera\nFor few seconds")
        self.my_pic = Image.open("/home/pi/Face-Recognition-Based-Attendance-System/face.jpg")
        self.resized = self.my_pic.resize((300,300),Image.ANTIALIAS)
        self.new_pic = ImageTk.PhotoImage(self.resized)
        self.my_label = Label(root, image = self.new_pic, bg = "white")
        self.my_label.place(x=530,y=180)
        self.after(800, self.TakeImages)
        
    def TakeImages(self):
        
        print("TI")
        val=usrdata.data
        x=val.split(" ")

        Id=x[2]#date
        print(Id)

        name=x[1]#name
        print(name)

    ##    UID=x[0]#UID
    ##    print(UID)
        
        #Id=(txt.get())
        #name=(txt2.get())
    
        cam = cv2.VideoCapture(0)
        harcascadePath = "haarcascade_frontalface_default.xml"
        detector=cv2.CascadeClassifier(harcascadePath)
        sampleNum=0
        while(True):
            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.3, 5)
            for (x,y,w,h) in faces:
                cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)        
                #incrementing sample number 
                sampleNum=sampleNum+1
                #saving the captured face in the dataset folder TrainingImage
                cv2.imwrite("TrainingImage/ "+name +"."+Id +'.'+ str(sampleNum) + ".jpg", gray[y:y+h,x:x+w])
                #display the frame
                winname = "img"
                cv2.namedWindow(winname)        # Create a named window
                cv2.moveWindow(winname, 500,200)  # Move it to (40,30)
                img = cv2.resize(img, (350,300))
                cv2.imshow('img',img)
                
            #wait for 100 miliseconds 
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
            # break if the sample number is morethan 100
            elif sampleNum>30:
                break
        cam.release()
        cv2.destroyAllWindows() 
        res = "Images Saved for ID : " + Id +" Name : "+ name
        row = [Id , name]
        print("csv0")
        with open('StudentDetails/StudentDetails.csv','a+') as csvFile:
            print("csv1")
            writer = csv.writer(csvFile)
            writer.writerow(row)
            print("csv2")
        csvFile.close()
        self.label.configure(text= res)
        print("TT")
        self.after(500, self.TrainImages)
   
    def TrainImages(self):
     
        recognizer = cv2.face.LBPHFaceRecognizer_create()#cv2.createLBPHFaceRecognizer()
        recognizer.read("TrainingImageLabel/Trainner.yml")
        harcascadePath = "haarcascade_frontalface_default.xml"
        faceCascade = cv2.CascadeClassifier(harcascadePath);    
        df=pd.read_csv("StudentDetails/StudentDetails.csv")

        self.my_pic = Image.open("/home/pi/Face-Recognition-Based-Attendance-System/tick.jpg")
        self.resized = self.my_pic.resize((300,300),Image.ANTIALIAS)
        self.new_pic = ImageTk.PhotoImage(self.resized)
        self.my_label = Label(root, image = self.new_pic, bg = "white")
        self.my_label.place(x=530,y=180)
          
        recognizer = cv2.face_LBPHFaceRecognizer.create()#recognizer = cv2.face.LBPHFaceRecognizer_create()#$cv2.createLBPHFaceRecognizer()
        harcascadePath = "haarcascade_frontalface_default.xml"
        detector =cv2.CascadeClassifier(harcascadePath)
        faces,Id = self.getImagesAndLabels("TrainingImage")
        recognizer.train(faces, np.array(Id))
        print("TI")
        recognizer.save("TrainingImageLabel/Trainner.yml")
        print("TI1")
        res = "Image Trained Successfully"#+",".join(str(f) for f in Id)
        self.label.configure(text= res)
       
        self.after(5000, self.step1)
          
    def getImagesAndLabels(self,path):
        #get the path of all the files in the folder
        imagePaths=[os.path.join(path,f) for f in os.listdir(path)] 
        #print(imagePaths)
        print("GIAL")
        #create empth face list
        faces=[]
        #create empty ID list
        Ids=[]
        print("GIAL1")
        #now looping through all the image paths and loading the Ids and the images
        for imagePath in imagePaths:
            #loading the image and converting it to gray scale
            pilImage=Image.open(imagePath).convert('L')
            #Now we are converting the PIL image into numpy array
            imageNp=np.array(pilImage,'uint8')
            #getting the Id from the image
            Id=int(os.path.split(imagePath)[-1].split(".")[1])
            # extract the face from the training image sample
            faces.append(imageNp)
            Ids.append(Id)      
        return faces,Ids


root = Tk()
root.configure(background='white')
app=App(root)
root.wm_title("Temperature Monitoring System")
root.geometry("1360x710+0+0")

my_pic = Image.open("/home/pi/Face-Recognition-Based-Attendance-System/logo.png")
resized = my_pic.resize((150,150),Image.ANTIALIAS)
new_pic = ImageTk.PhotoImage(resized)
my_label = Label(root, image = new_pic, bg = "white")
my_label.place(x=65,y=15)
 


root.mainloop()
