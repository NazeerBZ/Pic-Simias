from PyQt5 import QtCore, QtGui, QtWidgets # Import the PyQt5 module we'll need
import sys # We need sys so that we can pass argv to QApplication
import numpy as np
import cv2
import design
from BasicEdits import BasicEdits
from SmoothingSpatialFilter import SmoothingSpatialFilter
from SharpeningSpatialFilter import SharpeningSpatialFilter
from Thresholding import Thresholding
from ArithmeticOperation import ArithmeticOperation
from HistogramEqualization import HistogramEqualization
from MorphologicalTransformation import MorphologicalTransformation

class App(QtWidgets.QMainWindow, design.Ui_MainWindow):
        
    img = np.array([])
    path = ''
    
    def __init__(self):        
        super(App, self).__init__(parent=None) # now i can recieve parent
        self.setupUi(self) # This is defined in design.py file, It sets up layout and widgets that are defined

        pixmap = QtGui.QPixmap('./sys_img/defaultImage.jpg')
        self.label_image.setPixmap(pixmap.scaled(741,551))                   
        
        # initially all elements are hidden
        self.hideAllWidgets()
        
        #connecting to functions
        self.actionOpen.triggered.connect(self.openImage)
        self.actionSave.triggered.connect(self.saveImage)
        self.actionClose.triggered.connect(self.closeImage)
        
        # Instance of all classes and passing elements to their constructor
        self.BasicEdits = BasicEdits(self.zoomSlider,self.rotationDial, self.rangeRow1, self.rangeRow2, self.rangeCol1, self.rangeCol2, self.resizeByRow, self.resizeByCol)
        self.SmoothingSpatialFilter = SmoothingSpatialFilter(self.averageSlider, self.gaussianSlider, self.medianSlider, self.bilateralSlider)
        self.SharpeningSpatialFilter = SharpeningSpatialFilter(self.laplacianCheckbox, self.cannyCheckBox, self.sobelXcheckBox, self.sobelYcheckBox)        
        self.Thresholding = Thresholding(self.globalThresholdCheckbox, self.meanThresholdCheckbox, self.gaussianThresholdCheckbox)
        self.ArithmeticOperation = ArithmeticOperation()
        self.HistogramEqualization = HistogramEqualization(self.histogramEqlCheckBox, self.claheCheckBox)
        self.MorphologicalTransformation = MorphologicalTransformation(self.erosionSlider, self.dilationSlider, self.openingSlider, self.closingSlider)
        
        # Connecting ToolBox widget to a function             
        self.toolBox.itemSelectionChanged.connect(self.toolList);
        
    def hideAllWidgets(self):
        self.zoomLabel.hide()
        self.zoomSlider.setValue(0)
        self.zoomSlider.hide()
        self.rowLabel.hide()
        self.rangeRow1.hide()
        self.rangeRow2.hide()
        self.columnLabel.hide()
        self.rangeCol1.hide()
        self.rangeCol2.hide()
        self.rotateLabel.hide()
        self.rotationDial.hide()
        self.resizeHeightLabel.hide()
        self.resizeByRow.hide()
        self.resizeWidthLabel.hide()
        self.resizeByCol.hide()
        self.averageLabel.hide()
        self.averageSlider.setValue(1)
        self.averageSlider.hide()           
        self.gaussianLabel.hide()
        self.gaussianSlider.setValue(1)
        self.gaussianSlider.hide()
        self.medianLabel.hide()
        self.medianSlider.setValue(1)
        self.medianSlider.hide()
        self.bilateralLabel.hide()
        self.bilateralSlider.setValue(1)
        self.bilateralSlider.hide()
        self.sobelLabel.hide()
        self.sobelXcheckBox.hide()
        self.sobelXcheckBox.setChecked(False)
        self.sobelYcheckBox.hide()
        self.sobelYcheckBox.setChecked(False)
        self.cannyCheckBox.hide()
        self.cannyCheckBox.setChecked(False)
        self.laplacianCheckbox.hide()
        self.laplacianCheckbox.setChecked(False)
        self.globalThresholdCheckbox.hide()
        self.globalThresholdCheckbox.setChecked(False)
        self.meanThresholdCheckbox.hide()
        self.meanThresholdCheckbox.setChecked(False)
        self.gaussianThresholdCheckbox.hide()
        self.gaussianThresholdCheckbox.setChecked(False)
        self.cancelLinkButton.hide()
        self.histogramEqlCheckBox.setChecked(False)
        self.histogramEqlCheckBox.hide()
        self.claheCheckBox.setChecked(False)
        self.claheCheckBox.hide()
        self.erosionLabel.hide()
        self.erosionSlider.setValue(1)
        self.erosionSlider.hide()
        self.dilationLabel.hide()
        self.dilationSlider.setValue(1)
        self.dilationSlider.hide()
        self.openingLabel.hide()
        self.openingSlider.setValue(1)
        self.openingSlider.hide()
        self.closingLabel.hide()
        self.closingSlider.setValue(1)
        self.closingSlider.hide()
    
    def toolList(self):
        if self.img.size :
            selectedItem = self.toolBox.currentItem()
            for key in self.tools.keys():
                if key == selectedItem.text(0):
                    self.tools[key](self)
        else: print('choose an image')

    def readImage(self, directory):
        self.img = cv2.imread(directory)
        
    def writeImage(self, directory, img):
        cv2.imwrite(directory, img)
    
    def setImage(self):
         pixmap = QtGui.QPixmap('./sys_img/temp.jpg')         
         self.label_image.setPixmap(pixmap)
         
    
    def openImage(self):
       self.pickImage()
       if self.path:               
             image = cv2.imread(self.path)
             
             if  image.shape[1] > 741 and image.shape[0] > 551:
                 newImg = cv2.resize(image, (741,551))
                 self.writeImage('./sys_img/temp.jpg', newImg)             
                 pixmap = QtGui.QPixmap('./sys_img/temp.jpg') 
                 self.label_image.setPixmap(pixmap)
                 self.readImage('./sys_img/temp.jpg')
                 # some stuffs required to be done earlier
                 self.BasicEdits.adjustZoomRange(self.img)        
                 self.BasicEdits.adjustCropRange(self.img)               
             
             else:
                 self.writeImage('./sys_img/temp.jpg', image)
                 pixmap = QtGui.QPixmap('./sys_img/temp.jpg')
                 self.label_image.setPixmap(pixmap)
                 self.readImage('./sys_img/temp.jpg')                 
                 self.BasicEdits.adjustZoomRange(self.img)       
                 self.BasicEdits.adjustCropRange(self.img)
             
                       
    def saveImage(self):
        directory, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Save image', '', 'All Files (*);;jpg (*.jpg);;png (*.png)');
        if(directory):
            self.readImage('./sys_img/temp.jpg')
            self.writeImage(directory, self.img)

    def closeImage(self):      
        pixmap = QtGui.QPixmap('./sys_img/defaultImage.jpg')
        self.label_image.setPixmap(pixmap.scaled(741,551))
        self.img = np.array([])
        self.hideAllWidgets();
        
    def isGrayscaleImage(self):
        b,g,r = cv2.split(self.img)
        result1 = np.array_equal(b, g)
        result2 = np.array_equal(g, r)
        if result1 and result2:
           return True
        else: return False
    
    def pickImage(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self,'Pick image')
        self.path = path        
     
##############################################################################    
    def doZoom(self):
        self.hideAllWidgets()
        self.zoomLabel.move(10,30)
        self.zoomLabel.show()
        self.zoomSlider.show()
        self.zoomSlider.move(50,30)        
        self.zoomSlider.valueChanged.connect(lambda: self.BasicEdits.imgZoom(self.img, self.writeImage, self.setImage))

    def doRotate(self):
        self.hideAllWidgets()
        self.rotateLabel.move(10,50)
        self.rotateLabel.show()
        self.rotationDial.move(50,30)
        self.rotationDial.show()      
        self.rotationDial.valueChanged.connect(lambda: self.BasicEdits.imgRotate(self.img, self.writeImage, self.setImage));        
        
    def doCrop(self):
        self.hideAllWidgets()
        self.rowLabel.move(10,30)
        self.rowLabel.show()
        self.rangeRow1.move(10,60)
        self.rangeRow1.show()        
        self.rangeRow2.move(10,90)
        self.rangeRow2.show()
        self.columnLabel.move(90,30)
        self.columnLabel.show()    
        self.rangeCol1.move(90,60)
        self.rangeCol1.show()         
        self.rangeCol2.move(90,90)
        self.rangeCol2.show()
        
        self.rangeRow1.valueChanged.connect(lambda: self.BasicEdits.imgCrop(self.writeImage, self.setImage));
        self.rangeRow2.valueChanged.connect(lambda: self.BasicEdits.imgCrop(self.writeImage, self.setImage));
        self.rangeCol1.valueChanged.connect(lambda: self.BasicEdits.imgCrop(self.writeImage, self.setImage));
        self.rangeCol2.valueChanged.connect(lambda: self.BasicEdits.imgCrop(self.writeImage, self.setImage));
   
    def doResize(self):
        self.hideAllWidgets()
        self.resizeWidthLabel.move(10,40)
        self.resizeWidthLabel.show()
        self.resizeHeightLabel.move(90,40)
        self.resizeHeightLabel.show()
        self.resizeByCol.move(10,60)
        self.resizeByCol.show()
        self.resizeByRow.move(90,60)
        self.resizeByRow.show()
        
        self.resizeByRow.valueChanged.connect(lambda: self.BasicEdits.imgResize(self.img, self.writeImage, self.setImage))
        self.resizeByCol.valueChanged.connect(lambda: self.BasicEdits.imgResize(self.img, self.writeImage, self.setImage))
    
    def doGrayscale(self):
        self.hideAllWidgets()
        self.BasicEdits.grayscale(self.img, self.writeImage, self.setImage)
    
    def doAverage(self):
        self.hideAllWidgets()
        self.averageLabel.move(10,30)
        self.averageLabel.show()
        self.averageSlider.move(60,30)
        self.averageSlider.show()
        self.averageSlider.valueChanged.connect(lambda: self.SmoothingSpatialFilter.average(self.img, self.writeImage, self.setImage))        
    
    def doGaussian(self):
        self.hideAllWidgets()
        self.gaussianLabel.move(10,30)
        self.gaussianLabel.show()
        self.gaussianSlider.move(60,30)
        self.gaussianSlider.show()
        self.gaussianSlider.valueChanged.connect(lambda: self.SmoothingSpatialFilter.gaussian(self.img, self.writeImage, self.setImage))        
    
    def doMedian(self):
        self.hideAllWidgets()
        self.medianLabel.move(10,30)
        self.medianLabel.show()
        self.medianSlider.move(60,30)
        self.medianSlider.show()
        self.medianSlider.valueChanged.connect(lambda: self.SmoothingSpatialFilter.median(self.img, self.writeImage, self.setImage))
    
    def doBilateral(self):
        self.hideAllWidgets()
        self.bilateralLabel.move(10,30)
        self.bilateralLabel.show()
        self.bilateralSlider.move(60,30)
        self.bilateralSlider.show()
        self.bilateralSlider.valueChanged.connect(lambda: self.SmoothingSpatialFilter.bilateral(self.img, self.writeImage, self.setImage))
    
    def doLaplacian(self):        
        self.hideAllWidgets()
        if self.isGrayscaleImage():
            self.laplacianCheckbox.move(10,30)
            self.laplacianCheckbox.show()          
            self.laplacianCheckbox.stateChanged.connect(lambda: self.SharpeningSpatialFilter.laplacian(self.img, self.writeImage, self.setImage))   
        else:
           print('**Image must be grayscale**')
        
    def doCanny(self): 
        self.hideAllWidgets()
        if self.isGrayscaleImage():
            self.cannyCheckBox.move(10,30)
            self.cannyCheckBox.show()
            self.cannyCheckBox.stateChanged.connect(lambda: self.SharpeningSpatialFilter.canny(self.img, self.writeImage, self.setImage))        
        else:
            print('**Image must be grayscale**')
        
    def doSobel(self):
        self.hideAllWidgets()
        if self.isGrayscaleImage():
            self.sobelLabel.move(10,30)
            self.sobelLabel.show()
            self.sobelXcheckBox.move(10,60)
            self.sobelXcheckBox.show()
            self.sobelYcheckBox.move(90,60)
            self.sobelYcheckBox.show()
            self.sobelXcheckBox.stateChanged.connect(lambda: self.SharpeningSpatialFilter.sobelX(self.img, self.writeImage, self.setImage))
            self.sobelYcheckBox.stateChanged.connect(lambda: self.SharpeningSpatialFilter.sobelY(self.img, self.writeImage, self.setImage))
        else:
            print('**Image must be grayscale**')        
    
    def doBlending(self):
        self.hideAllWidgets()
        self.cancelLinkButton.move(10,30)
        self.cancelLinkButton.show()
        self.pickImage()
        img1 = self.img
        img2 = cv2.imread(self.path)        
        self.ArithmeticOperation.blending(img1, img2, self.writeImage, self.setImage)
        self.cancelLinkButton.clicked.connect(lambda: self.ArithmeticOperation.cancelEdited(self.img, self.writeImage, self.setImage))
    
    def doBitwise(self):
        self.hideAllWidgets()
        self.cancelLinkButton.move(10,30)
        self.cancelLinkButton.show()
        self.pickImage()
        img1 = self.img
        img2 = cv2.imread(self.path)        
        self.ArithmeticOperation.bitwise(img1, img2, self.writeImage, self.setImage)        
        self.cancelLinkButton.clicked.connect(lambda: self.ArithmeticOperation.cancelEdited(self.img, self.writeImage, self.setImage))
    
    def doHE(self):
        self.hideAllWidgets()
        if self.isGrayscaleImage():
            self.histogramEqlCheckBox.move(10,30)
            self.histogramEqlCheckBox.show()
            self.histogramEqlCheckBox.stateChanged.connect(lambda: self.HistogramEqualization.he(self.img, self.writeImage, self.setImage))            
        else:
             print('**Image must be grayscale**')
    
    def doCLAHE(self):
        self.hideAllWidgets()
        if self.isGrayscaleImage():
            self.claheCheckBox.move(10,30)
            self.claheCheckBox.show()
            self.claheCheckBox.stateChanged.connect(lambda: self.HistogramEqualization.clahe(self.img, self.writeImage, self.setImage))            
        else:
            print('**Image must be grayscale**')
    
    def doGlobalThreshold(self):
        self.hideAllWidgets()        
        if self.isGrayscaleImage():
             self.globalThresholdCheckbox.move(10,30)
             self.globalThresholdCheckbox.show()
             self.globalThresholdCheckbox.stateChanged.connect(lambda: self.Thresholding.globalThreshold(self.img, self.writeImage, self.setImage))        
        else:
           print('**Image must be grayscale**')
           
       
    def doMeanThreshold(self):
        self.hideAllWidgets()  
        if self.isGrayscaleImage():
            self.meanThresholdCheckbox.move(10,30)
            self.meanThresholdCheckbox.show()
            self.meanThresholdCheckbox.stateChanged.connect(lambda: self.Thresholding.meanThreshold(self.img, self.writeImage, self.setImage))
        else:
            print('**Image must be grayscale**')
        
        
    def doGaussianThreshold(self):
        self.hideAllWidgets()   
        if self.isGrayscaleImage():
            self.gaussianThresholdCheckbox.move(10,30)
            self.gaussianThresholdCheckbox.show()
            self.gaussianThresholdCheckbox.stateChanged.connect(lambda: self.Thresholding.gaussianThreshold(self.img, self.writeImage, self.setImage))
        else:
             print('**Image must be grayscale**')
    
    def doErosion(self):
        self.hideAllWidgets()
        if self.isGrayscaleImage():
            self.erosionLabel.move(10,30)
            self.erosionLabel.show()
            self.erosionSlider.move(60,30)
            self.erosionSlider.show()
            self.erosionSlider.valueChanged.connect(lambda: self.MorphologicalTransformation.erosion(self.img, self.writeImage, self.setImage))
           
        else:
            print('**Image must be grayscale**')
    
    def doDilation(self):
        self.hideAllWidgets()
        if self.isGrayscaleImage():
            self.dilationLabel.move(10,30)
            self.dilationLabel.show()
            self.dilationSlider.move(60,30)
            self.dilationSlider.show()
            self.dilationSlider.valueChanged.connect(lambda: self.MorphologicalTransformation.dilation(self.img, self.writeImage, self.setImage))            
        else:
            print('**Image must be grayscale**')
    
    def doOpening(self):
        self.hideAllWidgets()
        if self.isGrayscaleImage():
            self.openingLabel.move(10,30)
            self.openingLabel.show()
            self.openingSlider.move(60,30)
            self.openingSlider.show()
            self.openingSlider.valueChanged.connect(lambda: self.MorphologicalTransformation.opening(self.img, self.writeImage, self.setImage))            
        else:
            print('**Image must be grayscale**')
        
    def doClosing(self):
        self.hideAllWidgets()
        if self.isGrayscaleImage():
            self.closingLabel.move(10,30)
            self.closingLabel.show()
            self.closingSlider.move(60,30)
            self.closingSlider.show()
            self.closingSlider.valueChanged.connect(lambda: self.MorphologicalTransformation.closing(self.img, self.writeImage, self.setImage))            
        else:
            print('**Image must be grayscale**')
    
    
    tools = {
            'Zoom': doZoom,
            'Rotate': doRotate,
            'Crop': doCrop,
            'Resize': doResize,
            'Grayscale': doGrayscale,
            'Averaging': doAverage,
            'Gaussian': doGaussian,
            'Median': doMedian,
            'Bilateral': doBilateral,
            'Laplacian': doLaplacian,
            'Canny': doCanny,
            'Sobel': doSobel,
            'Blending': doBlending,
            'Bitwise': doBitwise,
            'HE': doHE,
            'CLAHE': doCLAHE,
            'Global Threshold': doGlobalThreshold,
            'Mean Threshold': doMeanThreshold,
            'Gaussian Threshold': doGaussianThreshold,
            'Erosion': doErosion,
            'Dilation': doDilation,
            'Opening': doOpening,
            'Closing': doClosing
            }
    
    
    
def main():
    app = QtWidgets.QApplication(sys.argv)  # A new instance of QApplication
    form = App()                        # We set the form to be our ExampleApp (design)
    form.show()                         # Show the form
    app.exec_()                         # and execute the app

if __name__ == '__main__':              # if we're running file directly and not importing it
    main()                              # run the main function

