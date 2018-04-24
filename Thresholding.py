from PyQt5 import QtCore, QtGui, QtWidgets # Import the PyQt5 module we'll need
import numpy as np
import cv2

class Thresholding:
    def __init__(self, globalThresholdCheckbox, meanThresholdCheckbox, gaussianThresholdCheckbox):
        self.globalThresholdCheckbox = globalThresholdCheckbox
        self.meanThresholdCheckbox = meanThresholdCheckbox
        self.gaussianThresholdCheckbox = gaussianThresholdCheckbox        
        
    def globalThreshold(self, img, writeImage, setImage):
        if self.globalThresholdCheckbox.isChecked():
            img = cv2.medianBlur(img, 5)
            ret , th = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
            writeImage('./sys_img/temp.jpg', th)
            setImage()  
        else:
            writeImage('./sys_img/temp.jpg', img)
            setImage()                
            
    def meanThreshold(self, img, writeImage, setImage):
        if self.meanThresholdCheckbox.isChecked():
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.medianBlur(img, 5)
            th = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2)
            writeImage('./sys_img/temp.jpg', th)
            setImage()  
        else:
            writeImage('./sys_img/temp.jpg', img)
            setImage()        
    
    def gaussianThreshold(self, img, writeImage, setImage):
        if self.gaussianThresholdCheckbox.isChecked():
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.medianBlur(img, 5)
            th = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
            writeImage('./sys_img/temp.jpg', th)
            setImage()  
        else:
            writeImage('./sys_img/temp.jpg', img)
            setImage()   
            
            
            