import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import PIL
import csv
import tensorflow as tf
from random import shuffle
import tflearn
from tqdm import tqdm
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

Train_Directory='Train'
Test_Directory='Test'
imageSize=244
ModelName="SportsClassification"
LR = 0.001
Sports_label_dict={
    'Basketball':0,
    'Football':1,
    'Rowing':2,
    'Swimming':3,
    'Tennis':4,
    'Yoga':5,
}

def createLabel(image_name):
    word_label = image_name.split('_')[0]
    if word_label == 'Basketball':
        return Sports_label_dict['Basketball']
    elif word_label == 'Football':
        return Sports_label_dict['Football']
    elif word_label == 'Rowing':
        return Sports_label_dict['Rowing']
    elif word_label == 'Swimming':
        return Sports_label_dict['Swimming']
    elif word_label == 'Tennis':
        return Sports_label_dict['Tennis']
    else:
        return Sports_label_dict['Yoga']

def createTrainData(Train_Directory):
    training_data= []
    for img in tqdm(os.listdir(Train_Directory)):
        path = os.path.join(Train_Directory, img)
        img_data = cv2.imread(path)
        img_data = cv2.resize(img_data, (imageSize, imageSize))
        y=createLabel(img)
        training_data.append([np.array(img_data), y])
    shuffle(training_data)
    np.save('train_data.npy', training_data)
    return training_data

def createTestData(Test_Directory):
    testing_data=[]
    for img in tqdm(os.listdir(Test_Directory)):
        path = os.path.join(Test_Directory, img)
        img_data = cv2.imread(path)
        img_data = cv2.resize(img_data, (imageSize, imageSize))
        img_data= img_data.reshape(-1,imageSize, imageSize, 3)
        testing_data.append([np.array(img_data)])
    return testing_data  

if (os.path.exists('train_data.npy')): # If you have already created the dataset:
    Trainingset =np.load('train_data.npy',allow_pickle=True)
else: # If dataset is not created:
    Trainingset = createTrainData(Train_Directory)
    
if (os.path.exists('test_data.npy')):
    TestingSet =np.load('test_data.npy',allow_pickle=True)
else:
    TestingSet = createTestData(Test_Directory)



X_train = np.array([i[0] for i in Trainingset]).reshape(-1, imageSize, imageSize, 3)
y_train = [i[1] for i in Trainingset]

X_test = np.array([i[0] for i in TestingSet]).reshape(-1, imageSize, imageSize, 1)



X_train=np.array(X_train)
y_train=np.array(y_train)
x_train_scaled=X_train/255

data_augmentation = keras.Sequential(
  [
    layers.experimental.preprocessing.RandomFlip("horizontal", 
                                                 input_shape=(imageSize, 
                                                              imageSize,
                                                              3)),
    layers.experimental.preprocessing.RandomRotation(0.2),
    layers.experimental.preprocessing.RandomZoom(0.1),
  ]
)

num_classes = 6

model = Sequential([
  #data_augmentation,
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(2,strides=2),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(2,strides=2),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(2,strides=2),
  layers.Conv2D(128, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(2,strides=2),
  layers.Conv2D(256, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(2,strides=2),
  layers.Conv2D(512, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(2,strides=4),
  layers.Conv2D(1024, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(2,strides=4),
  layers.Dropout(0.2),
  layers.Flatten(),
  #layers.Dense(128),
  layers.Dense(num_classes, activation='softmax')
])


model.compile(optimizer='adamax',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
              
model.fit(x_train_scaled, y_train, epochs=30)  


rows=[]
for img in tqdm(os.listdir(Test_Directory)):
    print(img)
    path = os.path.join(Test_Directory, img)
    img_data = cv2.imread(path)
    test_img = cv2.resize(img_data, (244, 244))
    #print(test_img.shape)
    test_img = test_img.reshape(-1,244, 244, 3)
    #print(test_img.shape)
    prediction = model.predict([test_img])
    #print(prediction)
    max_value = max(prediction[0])
    #print(max_value)
    #print(prediction[0])
    max_index = prediction[0].tolist().index(max_value)
    #print(max_index)
    A_prediction=[]
    A_prediction.append(img)
    A_prediction.append(max_index)
    rows.append(A_prediction)
#rows.append(listt)
filename="predictions.csv"    
# writing to csv file 
fields = ['image_name', 'label']

with open(filename, 'w',newline='') as csvfile: 
    # creating a csv writer object 
    csvwriter = csv.writer(csvfile) 
        
    # writing the fields 
    csvwriter.writerow(fields) 
        
    # writing the data rows 
    csvwriter.writerows(rows)