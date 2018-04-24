from PyQt5 import QtCore, QtGui, QtWidgets # Import the PyQt5 module we'll need
import numpy as np
import cv2

class HistogramEqualization:
    def __init__(self, histogramEqlCheckBox, claheCheckBox):
        self.histogramEqlCheckBox = histogramEqlCheckBox
        self.claheCheckBox = claheCheckBox
        
    def he(self, img, writeImage, setImage):
        if self.histogramEqlCheckBox.isChecked():
            # input grayscale image has rgb intensity values in each pixel
            # but some functions required grayscale image to be just with
            # one intensity value in each pixel 
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            equ = cv2.equalizeHist(img)
            #res = np.hstack((grayscaleImg, equ))
            writeImage('./sys_img/temp.jpg', equ)
            setImage()
        else:
            writeImage('./sys_img/temp.jpg', img)
            setImage()
        
        
    
    def clahe(self, img, writeImage, setImage):
        if self.claheCheckBox.isChecked():
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit = 2.0, tileGridSize=(8,8))
            newImg = clahe.apply(img);
            writeImage('./sys_img/temp.jpg', newImg)
            setImage()
        else:
            writeImage('./sys_img/temp.jpg', img)
            setImage()
            
        
        
