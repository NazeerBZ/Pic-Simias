from PyQt5 import QtCore, QtGui, QtWidgets # Import the PyQt5 module we'll need
import numpy as np
import cv2
import sys
import os
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras import backend as K
K.set_image_dim_ordering('th') 


class face_recognition:
    def __init__(self, label_image):
        self.label_image = label_image
        self.subjects = ['Nazeer', 'Messi', 'Suarez', 'Neymar']        
        self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.stop = False
        x_train, y_train = self.create_trainset('./dataset')
        self.face_recognizer.train(x_train, np.array(y_train))
        
    def detect_faces(self, colored_img, scalerFactor=1.1):    
        gray_img = cv2.cvtColor(colored_img, cv2.COLOR_BGR2GRAY)    
        face_cascade = cv2.CascadeClassifier('pretrained_models/haarcascade_frontalface_alt.xml')
        faces = face_cascade.detectMultiScale(gray_img, scaleFactor=scalerFactor, minNeighbors=5)
    
        if len(faces) == 0:
            return None, None
    
        grayscale_faces = []
        faces_pos = []
        for face in faces:
            (x,y,w,h) = face
            grayscale_faces.append(gray_img[y:y+w, x:x+h]) # Image ROI that represents face
            faces_pos.append(face)       
            
        return grayscale_faces, faces_pos
    
    #Call if need to prepare dataset for faces
    def prepare_dataset(self,dataset_path):
        dirsNameLs = os.listdir(dataset_path)
        for dir_name in dirsNameLs:
            subject_path = dataset_path + '/' + dir_name
            imgsNameLs = os.listdir(subject_path)
            for image_name in imgsNameLs:
                image_path = subject_path + '/' + image_name
                image = cv2.imread(image_path)
                gs_faces, faces_pos = self.detect_faces(image)
                if gs_faces is not None:           
                    for face in gs_faces:
                        sample = cv2.resize(face, (224,224))
                        cv2.imwrite(image_path,sample)

    # just to make faces dataset
    #prepare_dataset('./image_for_dataset') 
    
    def create_trainset(self, dataset_path):    
        dirsNameLs = os.listdir(dataset_path)
        x_train = [] # contains all faces in dataset in linear order
        y_train = [] # containes all labels
        for dir_name in dirsNameLs:
            label = int(dir_name)
            subject_path = dataset_path + '/' + dir_name
            imgsNameLs = os.listdir(subject_path)
            for image_name in imgsNameLs:
                image_path = subject_path + '/' + image_name
                image = cv2.imread(image_path,0)
                x_train.append(image)
                y_train.append(label)
            
        return x_train, y_train

    def drawRectangleWithText(self, frame, rect, text):
        (x,y,w,h) = rect
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(frame, text, (x,y-5), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

    def predict(self, frame):
        counter = 0
        gs_faces, faces_pos = self.detect_faces(frame)
        if gs_faces != None:
            print('Detected Faces: ', len(gs_faces))
            print('Faces Position', faces_pos)
            for face in gs_faces:
                label = self.face_recognizer.predict(face)
                print('(label, acc):',label)
#                text = subjects[label[0]] + ' ' + str('{0:.2f}'.format(label[1]))
                text = self.subjects[label[0]]
                rectangle = faces_pos[counter]
                self.drawRectangleWithText(frame, rectangle, text)   
                counter +=1
        return frame


    def stopRecognizer(self):
        self.stop = True

    def startRecognizerForWebcam(self):        
        cap = cv2.VideoCapture(0)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('./videos/output.avi', fourcc, 20.0, (640,480))
                
        while(cap.isOpened()):
            # Capture frame-by-frame
            ret, frame = cap.read()
            frame = self.predict(frame)            
            # Display the resulting frame            
            cv2.imshow('Face_Recognition', frame)
            out.write(frame)
#            frame = QtGui.QImage(frame, frame.shape[1], frame.shape[0], QtGui.QImage.Format_RGB888)
#            pixmap = QtGui.QPixmap.fromImage(frame)
#            self.label_image.setPixmap(pixmap.scaled(741,551))            
            cv2.waitKey(1)
            if self.stop == True:
                break
        # When everything is done, release the capture
        cap.release()
        cv2.destroyAllWindows()
    
    def startRecognizerForVideo(self):
        cap = cv2.VideoCapture('./videos/shortclip.mp4')
        
        while(cap.isOpened()):
            ret,frame = cap.read()
            frame = self.predict(frame)              
            cv2.imshow('Face_Recognition', frame)
            cv2.waitKey(1)
            if self.stop == True:
                break
        cap.release()
        cv2.destroyAllWindows()
    
    def startRecognizerOnImage(self, img, writeImage, setImage):
        output_img = self.predict(img)
        writeImage('./sys_img/temp.jpg', output_img)
        setImage()
    
    def onWebcam(self):
        self.startRecognizerForWebcam()
    
    def onVideo(self):
        self.startRecognizerForVideo()    
    
    def onImage(self, img, writeImage, setImage):
        self.startRecognizerOnImage(img, writeImage, setImage)
       

























