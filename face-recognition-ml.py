#OpenCV module
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np


subjects = ['Messi', 'Neymar', 'Suarez']

def detect_faces(colored_img, scalerFactor=1.1):
    
    gray_img = cv2.cvtColor(colored_img, cv2.COLOR_BGR2GRAY)    
    face_cascade = cv2.CascadeClassifier('trained_models/haarcascade_frontalface_alt.xml')
    faces = face_cascade.detectMultiScale(gray_img, scaleFactor=scalerFactor, minNeighbors=5)
    
    if len(faces) == 0:
        return None, None
    
    grayscale_faces = []
    faces_pos = []
    for face in faces:
        (x,y,w,h) = face
        grayscale_faces.append(gray_img[y:y+w, x:x+h])
        faces_pos.append(face)
        
#    (x,y,w,h) = faces[0]
    return grayscale_faces, faces_pos
    

def prepare_dataset(dataset_path):
    dirsNameLs = os.listdir(dataset_path)
    for dir_name in dirsNameLs:
        subject_path = dataset_path + '/' + dir_name
        imgsNameLs = os.listdir(subject_path)
        for image_name in imgsNameLs:
            image_path = subject_path + '/' + image_name
            image = cv2.imread(image_path)
            gs_faces, faces_pos = detect_faces(image)
            if gs_faces is not None:           
                for face in gs_faces:
                    sample = cv2.resize(face, (224,224))
                    cv2.imwrite(image_path,sample)

# just to make grayscale faces dataset
#prepare_dataset('./dataset') 

def prepare_training_set(dataset_path):
    
    dirsNameLs = os.listdir(dataset_path)
    faces = []
    labels = []
    for dir_name in dirsNameLs:
        label = int(dir_name)
        subject_path = dataset_path + '/' + dir_name
        imgsNameLs = os.listdir(subject_path)
        for image_name in imgsNameLs:
            image_path = subject_path + '/' + image_name
            image = cv2.imread(image_path,0)
            faces.append(image)
            labels.append(label)
            
    return faces, labels
                        

#faces, labels = prepare_training_set('./dataset')
#
#face_recognizer = cv2.face.LBPHFaceRecognizer_create()
#face_recognizer.train(faces, np.array(labels))

def drawRectangleWithText(img, rect, text):
    (x,y,w,h) = rect
    cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
    cv2.putText(img, text, (x,y-5), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
    

def predict(test_img):
    counter = 0
    gs_faces, faces_pos = detect_faces(test_img)
    print('Detected Faces: ', len(gs_faces))
    print('Faces Position', faces_pos)
    for face in gs_faces:
        label = face_recognizer.predict(face)
        print('(label, acc):',label)
        text = subjects[label[0]] + ' ' + str('{0:.2f}'.format(label[1]))
        rectangle = faces_pos[counter]
        drawRectangleWithText(test_img, rectangle, text)   
        counter +=1
    return test_img


#testImage = cv2.imread('./testdata/test3.jpg')
#predicted_img =  predict(testImage)
##cv2.namedWindow('Output', cv2.WINDOW_NORMAL)
#cv2.imshow('Output', predicted_img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
