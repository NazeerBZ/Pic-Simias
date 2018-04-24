from PyQt5 import QtCore, QtGui, QtWidgets # Import the PyQt5 module we'll need
import numpy as np
import cv2

class SmoothingSpatialFilter:
    def __init__(self, averageSlider, gaussianSlider, medianSlider, bilateralSlider):
      self.averageSlider = averageSlider
      self.gaussianSlider = gaussianSlider
      self.medianSlider = medianSlider
      self.bilateralSlider = bilateralSlider
    
    def average(self, img, writeImage, setImage):
#        kernal = np.ones((5,5))/25
#        newImg = cv2.filter2D(img, -1, kernal)
#        print(self.averageSlider.value())
        newImg = cv2.blur(img, (self.averageSlider.value(),self.averageSlider.value()))
        writeImage('./sys_img/temp.jpg', newImg);
        setImage();        
    
    def gaussian(self, img, writeImage, setImage):
#        print(self.gaussianSlider.value())
        newImg = cv2.GaussianBlur(img,(self.gaussianSlider.value(),self.gaussianSlider.value()),0);
        writeImage('./sys_img/temp.jpg', newImg);
        setImage();
    
    def median(self, img, writeImage, setImage):
#        print(self.medianSlider.value())
        newImg = cv2.medianBlur(img, self.medianSlider.value());
        writeImage('./sys_img/temp.jpg', newImg);
        setImage();
    
    def bilateral(self, img, writeImage, setImage):
#        print(self.bilateralSlider.value())
        newImg = cv2.bilateralFilter(img, self.bilateralSlider.value(),85,85)
        writeImage('./sys_img/temp.jpg', newImg);
        setImage();