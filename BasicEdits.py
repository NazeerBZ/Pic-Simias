from PyQt5 import QtCore, QtGui, QtWidgets # Import the PyQt5 module we'll need
import numpy as np
import cv2

class BasicEdits:
    def __init__(self, zoomSlider, rotationDial, rangeRow1, rangeRow2, rangeCol1, rangeCol2, resizeByRow, resizeByCol):
        self.zoomSlider = zoomSlider;
        self.rotationDial = rotationDial
        self.rangeRow1 = rangeRow1
        self.rangeRow2 = rangeRow2
        self.rangeCol1 = rangeCol1
        self.rangeCol2 = rangeCol2
        self.resizeByRow = resizeByRow
        self.resizeByCol = resizeByCol
    
    def adjustZoomRange(self, img):
        self.img = img;
#        self.zoomSlider.setMaximum(self.img.shape[0])
        if self.img.shape[0] > self.img.shape[1]: # is rows more then columns
               self.zoomSlider.setMaximum(self.img.shape[1]-30) # not to start from zero due possibility of going into negative
        else: # is columns more then rows
               self.zoomSlider.setMaximum(self.img.shape[0]-30) # not to start from zero due possibility of going into negative
               
    def imgZoom(self, img ,writeImage, setImage):
       self.img = img
        # (388, 647, 3) This means that the image has 388 rows, 647 columns, and 
        # 3 channels (the RGB components) but when working with images we normally specify 
        # images in terms of width x height Looking at the shape of the matrix, we may 
        # think that our image is 388 pixels wide and 647 pixels tall. However, this would be 
        # incorrect. Our image is actually 647 pixels wide and 388 pixels tall according to OpenCV
#       height = self.img.shape[0] # columns a/c to opencv
#       width = self.img.shape[1] # rows a/c to opencv
#       imgScale1 = self.zoomSlider.value()*300 # adjusting as width of label
#       imgScale2 = self.zoomSlider.value()*590 # adjusting as height of label
#          
#       newX, newY = imgScale2/width, imgScale1/height        
       rows, cols = self.img.shape[0]-self.zoomSlider.value() ,self.img.shape[1]-self.zoomSlider.value()
       
       newImg = cv2.resize(self.img, (int(cols),int(rows)), interpolation=cv2.INTER_LINEAR)
#          
       writeImage('./sys_img/temp.jpg', newImg)        
       setImage()
    
    def imgRotate(self, img, writeImage, setImage):
        self.img = img
        rows, cols = self.img.shape[:2]
        center = (cols/2, rows/2)
        M = cv2.getRotationMatrix2D(center, -self.rotationDial.value(), 1)
        rotated = cv2.warpAffine(self.img, M, (cols, rows))
        
        writeImage('./sys_img/temp.jpg', rotated);
        setImage();
    
    def adjustCropRange(self, img):
        self.img = img
        self.rangeRow1.setMaximum(self.img.shape[0])
        self.rangeRow2.setMaximum(self.img.shape[0])
        self.rangeCol1.setMaximum(self.img.shape[1])
        self.rangeCol2.setMaximum(self.img.shape[1])
    
    def imgCrop(self, writeImage, setImage):
        cropped = self.img[int(self.rangeRow1.value()):int(self.rangeRow2.value()), int(self.rangeCol1.value()):int(self.rangeCol2.value())];
        writeImage('./sys_img/temp.jpg', cropped)
        setImage();
        
        
    def imgResize(self, img, writeImage, setImage):
        height = self.resizeByRow.value()
        width = self.resizeByCol.value()
        resizedImg = cv2.resize(img, (width, height))
        writeImage('./sys_img/temp.jpg', resizedImg)
        setImage()
    
    def grayscale(self, img, writeImage, setImage):
        grayscaleImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        writeImage('./sys_img/temp.jpg', grayscaleImage)
        setImage()
        
        
        
        
        