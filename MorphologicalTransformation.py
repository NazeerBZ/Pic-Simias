from PyQt5 import QtCore, QtGui, QtWidgets # Import the PyQt5 module we'll need
import numpy as np
import cv2

#Morphological transformations are some simple operations based on the image shape. 
#It is normally performed on binary images. It needs two inputs, one is our original image, 
#second one is called structuring element or kernel. Two basic morphological operators are 
#Erosion and Dilation. Then its variant forms like Opening, Closing, Gradient etc

class MorphologicalTransformation:
    def __init__(self, erosionSlider, dilationSlider, openingSlider, closingSlider):
        self.erosionSlider = erosionSlider 
        self.dilationSlider = dilationSlider 
        self.openingSlider = openingSlider
        self.closingSlider = closingSlider
    
    def erosion(self, img, writeImage, setImage):
#        The kernel slides through the image (as in 2D convolution). A pixel in the original 
#        image (either 1 or 0) will be considered 1 only if all the pixels under the kernel is 1,
#        otherwise it is eroded (made to zero). So what happends is that, all the pixels near 
#        boundary will be discarded depending upon the size of kernel. So the thickness or 
#        size of the foreground object decreases or simply white region decreases in the image. 
#        It is useful for removing small white noises         
        kernel = np.ones((self.erosionSlider.value(),self.erosionSlider.value()), dtype='uint8')
        erosion = cv2.erode(img, kernel, iterations = 1)
        writeImage('./sys_img/temp.jpg', erosion)
        setImage()
    
    def dilation(self, img, writeImage, setImage):
#        It is just opposite of erosion. Here, a pixel element is ‘1’ if atleast one pixel
#        under the kernel is ‘1’. So it increases the white region in the image or size of 
#        foreground object increases. Normally, in cases like noise removal, erosion is followed 
#        by dilation. Because, erosion removes white noises, but it also shrinks our object. 
#        So we dilate it. Since noise is gone, they won’t come back, but our object area 
#        increases.
         kernel = np.ones((self.dilationSlider.value(),self.dilationSlider.value()), dtype='uint8')
         dilation = cv2.dilate(img, kernel, iterations = 1)
         writeImage('./sys_img/temp.jpg', dilation)
         setImage()
         
    def opening(self, img, writeImage, setImage):
#        Opening is just another name of erosion followed by dilation. 
        kernel = np.ones((self.openingSlider.value(), self.openingSlider.value()), dtype='uint8')
        opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        writeImage('./sys_img/temp.jpg', opening)
        setImage()
        
    def closing(self, img, writeImage, setImage):
#        Closing is reverse of Opening, Dilation followed by Erosion. 
        kernel = np.ones((self.closingSlider.value(), self.closingSlider.value()), dtype='uint8')
        closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        writeImage('./sys_img/temp.jpg', closing)
        setImage()
        
        
        
        