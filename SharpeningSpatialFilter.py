from PyQt5 import QtCore, QtGui, QtWidgets # Import the PyQt5 module we'll need
import numpy as np
import cv2
from matplotlib import pyplot as plt

class SharpeningSpatialFilter:
    def __init__(self, laplacianCheckbox, cannyCheckBox, sobelXcheckBox, sobelYcheckBox):
        self.laplacianCheckbox = laplacianCheckbox
        self.cannyCheckBox = cannyCheckBox
        self.sobelXcheckBox = sobelXcheckBox
        self.sobelYcheckBox = sobelYcheckBox
        
    def laplacian(self, img, writeImage, setImage):
        if self.laplacianCheckbox.isChecked():
            print('Laplacian Checked')                
            laplacian = cv2.Laplacian(img, cv2.CV_8U)
              
            writeImage('./sys_img/temp.jpg', laplacian)
            setImage()
        else:
            print('Laplacian unChecked')
            writeImage('./sys_img/temp.jpg', img)
            setImage()
            

    def canny(self, img, writeImage, setImage):
        if self.cannyCheckBox.isChecked():
            print('Canny Checked')
            canny = cv2.Canny(img, 100, 200);
            
            writeImage('./sys_img/temp.jpg', canny)
            setImage()
        else:
            print('canny unChecked')
            writeImage('./sys_img/temp.jpg', img)
            setImage()

    def sobelX(self, img, writeImage, setImage):
        
        if self.sobelXcheckBox.isChecked():    
            sobelX = cv2.Sobel(img,cv2.CV_8U,1,0,ksize=3)
            
            self.sobelYcheckBox.setChecked(False)
            
            writeImage('./sys_img/temp.jpg', sobelX)
            setImage()
        else:
            writeImage('./sys_img/temp.jpg', img)
            setImage()
        
            
    def sobelY(self, img, writeImage, setImage):
        if self.sobelYcheckBox.isChecked():
            sobelY = cv2.Sobel(img,cv2.CV_8U,0,1,ksize=3)
            
            self.sobelXcheckBox.setChecked(False)
            
            writeImage('./sys_img/temp.jpg', sobelY)
            setImage()
        else:
            writeImage('./sys_img/temp.jpg', img)
            setImage()
        
        
            
    