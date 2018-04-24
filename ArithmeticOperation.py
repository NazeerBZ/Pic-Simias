from PyQt5 import QtCore, QtGui, QtWidgets # Import the PyQt5 module we'll need
import numpy as np
import cv2

class ArithmeticOperation:
        
    def blending(self, img1, img2, writeImage, setImage):
        resizedImg2 = cv2.resize(img2, (img1.shape[1],img1.shape[0]))
        blended = cv2.addWeighted(img1, 0.7, resizedImg2, 0.3, 0)
        writeImage('./sys_img/temp.jpg', blended)
        setImage()
    
    def bitwise(self, img1, img2, writeImage, setImage):
        copyOrginal = np.copy(img1)
        roi = copyOrginal[0:img2.shape[0], 0:img2.shape[1]]
        img2ToGray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(img2ToGray, 10, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
        img1_bg = cv2.bitwise_and(roi, roi, mask = mask_inv)
        img2_bg = cv2.bitwise_and(img2, img2, mask = mask)
        dst = cv2.add(img1_bg, img2_bg)
        copyOrginal[0:img2.shape[0], 0:img2.shape[1]] = dst
        
        writeImage('./sys_img/temp.jpg', copyOrginal)
        setImage()
#        print(ret)
#        cv2.imshow('mask', mask)
#        cv2.imshow('mask_inv', mask_inv)
#        cv2.imshow('img1_bg', img1_bg)
#        cv2.imshow('img2_bg', img2_bg)
#        cv2.imshow('dst', dst)
#        cv2.waitKey(0)
#        cv2.destroyAllWindows()
        
    def cancelEdited(self, img, writeImage, setImage):
        print('cancel')
        writeImage('./sys_img/temp.jpg', img)
        setImage()
