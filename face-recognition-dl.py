import cv2
import os
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.utils import np_utils
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras import backend as K
K.set_image_dim_ordering('th') 

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

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
      

def create_dataset(dataset_path):
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
#create_dataset('./dataset') 


def prepare_dataset(dataset_path):    
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

                        
X, Y = prepare_dataset('./dataset')
x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.33, random_state=seed)

# make these sets as numpy array
x_train  = np.array(x_train)
x_test = np.array(x_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

x_train = x_train.reshape(x_train.shape[0], 1, x_train.shape[1], x_train.shape[2])
x_test = x_test.reshape(x_test.shape[0], 1, x_test.shape[1], x_test.shape[2])

##Normalize
x_train = x_train / 255
x_test  = x_test / 255

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

#print(y_test[1])
#plt.imshow(x_test[1,0], cmap='gray')

model = Sequential()
model.add(Conv2D(30, (5,5), input_shape=(1,224,224), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))
model.add(Conv2D(15, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=2)

scores = model.evaluate(x_test, y_test)
print('%.2f%%' % (scores[1]*100))

pred = model.predict(x_test)
y_pred = [int(x[0]) for x in pred]
print(y_pred)


#def drawRectangleWithText(img, rect, text):
#    (x,y,w,h) = rect
#    cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
#    cv2.putText(img, text, (x,y-5), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
#    

