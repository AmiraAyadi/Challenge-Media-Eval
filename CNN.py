#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 18:06:22 2017

@author: AntoineP
"""

import os
import cv2
import matplotlib.pyplot as plt
from scipy.misc import imread
import numpy as np
import pickle

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation,Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
import matplotlib
import matplotlib.pyplot as plt
import scipy
import scipy.ndimage
import numpy as np
from keras import backend as K
import keras
import re

#Importation du train
pathTrain = './data/train/'
pathVideos = os.listdir('./data/train/')
pathVideos.remove('.DS_Store')


X_train = []
y_train = []

with open('./data/train.pkl', 'rb') as f:
    dicoT = pickle.load(f)
    
dicoTrain = dict()    
for i in range(len(dicoT[0])):
    dicoTrain[re.sub(r'.*[0-9]\/(.*)',r'\1',dicoT[0][i])] = dicoT[1][i]
    

for elem in pathVideos:
    pathImages = os.listdir(pathTrain + elem )
    for i in range(len(pathImages)):
        image = pathImages[i]
        img = imread(pathTrain + elem +'/'+ image,'L')
        X_train.append(img)
        y_train.append(dicoTrain[elem])


    
    
from sys import getsizeof
#Importation du test
pathtest = './data/test/'
pathVideos = os.listdir('./data/test/')


X_test = []
y_test = []

with open('./data/test.pkl', 'rb') as f:
    dicoT2 = pickle.load(f)
    
dicotest = dict()    
for i in range(len(dicoT2[0])):
    dicotest[re.sub(r'.*[0-9]\/(.*)',r'\1',dicoT2[0][i])] = dicoT2[1][i]
    

for elem in pathVideos:
    pathImages = os.listdir(pathtest + elem ) 
    image = pathImages[3]
    img = imread(pathtest + elem +'/'+ image,'L')
    X_test.append(img)
    y_test.append(dicotest[elem])



    

#Pre-processing
    
X_train = np.array(X_train).astype('float32')
X_train /= 255

X_test = np.array(X_test).astype('float32')
X_test /= 255
nb_classes = 8

#On change les classes par un entier de 0 à 7
y_train = np.array(y_train).astype('int64')
y_train[y_train ==1001] = 0
y_train[y_train ==1009] = 1
y_train[y_train ==1011] = 2
y_train[y_train ==1013] = 3
y_train[y_train ==1014] = 4
y_train[y_train ==1016] = 5
y_train[y_train ==1017] = 6
y_train[y_train ==1019] = 7
img_rows, img_cols = 60, 80
X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)



y_test = np.array(y_test).astype('int64')
y_test[y_test ==1001] = 0
y_test[y_test ==1009] = 1
y_test[y_test ==1011] = 2
y_test[y_test ==1013] = 3
y_test[y_test ==1014] = 4
y_test[y_test ==1016] = 5
y_test[y_test ==1017] = 6
y_test[y_test ==1019] = 7
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)


#On "dummifie" les catégories
y_train = np_utils.to_categorical(y_train, nb_classes)
y_test = np_utils.to_categorical(y_test, nb_classes)


#Model


modelPlus = Sequential()
modelPlus.add(Conv2D(16, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(60,80,1)))
modelPlus.add(MaxPooling2D(pool_size=(2, 2)))
modelPlus.add(Conv2D(16,kernel_size=(3, 3),
                 activation='relu'))
modelPlus.add(MaxPooling2D(pool_size=(2, 2)))

modelPlus.add(Flatten())
modelPlus.add(Dense(256, activation='relu'))
modelPlus.add(Dense(64, activation='relu'))
#model.add(Dropout(0.5))
modelPlus.add(Dense(8, activation='softmax'))

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
modelPlus.compile(loss='categorical_crossentropy',metrics=['accuracy'],  optimizer=sgd)

modelPlus.summary()

batch_size = 256
epochs=100
modelPlus.fit(X_train, y_train, batch_size=batch_size, epochs=epochs,  verbose=1, validation_data=(X_test, y_test))

loss,acc = modelPlus.evaluate(X_test, y_test,  verbose=0)
index=800
print('The accuracy on the test set is ',(acc*100),'%')
plot_mnist_digit(X_test_mat[index])
cl=model.predict_classes(X_test_mat[index].reshape((1,28,28,1)))


print("le chiffre reconnu est: ", cl[0])
print("le chiffre à reconnaitre  est: ", np.argmax(y_test_cat[index]))




