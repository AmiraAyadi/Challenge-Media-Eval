#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 11:27:29 2017

@author: AntoineP
"""
import pickle
from keras.preprocessing import image
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model, load_model
from keras.layers import Input
import numpy as np
from keras.applications.xception import Xception, preprocess_input
import re
    
#On importe le modèle CNN Xception entrainé sur imagenet et on enleve le dernier layer qui permet de choisir
# la classe (ici catégorie d'image de imagenet) pour uniquement garder les features créées par le modèle permettant à l'algo
#de prendre une décision sur la classification

base_model = Xception(weights='imagenet')
model = Model(input=base_model.input, output=base_model.get_layer('avg_pool').output)
image_size = (100, 100)

pathTrain = './created_data2/data/train/'
pathVideos = os.listdir('./created_data2/data/train/')
pathVideos.remove('.DS_Store')

X_train = []
y_train = []

with open('./created_data2/data/train.pkl', 'rb') as f:
    dicoT = pickle.load(f)
    
dicoTrain = dict()    
for i in range(len(dicoT[0])):
    dicoTrain[re.sub(r'.*[0-9]\/(.*)',r'\1',dicoT[0][i])] = dicoT[1][i]

    
for elem in pathVideos:
    pathImages = os.listdir(pathTrain + elem )
    if '.DS_Store' in pathImages:
        pathImages.remove('.DS_Store')
    for i in range(len(pathImages)):
        imageP = pathImages[i]
        img = image.load_img(pathTrain + elem +'/'+ imageP, target_size=(100, 100))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        feature = model.predict(x)
        X_train.append(feature)
        y_train.append(dicoTrain[elem])
        
        

#Idem sur les données test
pathTest = './created_data2/data/test/'
pathVideos = os.listdir('./created_data2/data/test/')
pathVideos.remove('.DS_Store')

X_test = []
y_test = []

with open('./created_data2/data/test.pkl', 'rb') as f:
    dicoT = pickle.load(f)
    
dicoTest = dict()    
for i in range(len(dicoT[0])):
    dicoTest[re.sub(r'.*[0-9]\/(.*)',r'\1',dicoT[0][i])] = dicoT[1][i]

c=0
for elem in pathVideos:
    pathImages = os.listdir(pathTest + elem )
    c+=1
    for i in range(len(pathImages)):
        imageP = pathImages[i]
        img = image.load_img(pathTest + elem +'/'+ imageP, target_size=(100, 100))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        feature = model.predict(x)
        X_test.append(feature)
        y_test.append(dicoTest[elem])
        
        
        

#Sauvegarde des objets en local
with open('./created_data2/data/X_train.pkl', 'wb') as f:
    pickle.dump([X_train],f)
with open('./created_data2/data/X_test.pkl', 'wb') as f:
    pickle.dump([X_test],f)
with open('./created_data2/data/y_train.pkl', 'wb') as f:
    pickle.dump([y_train],f)
with open('./created_data2/data/y_test.pkl', 'wb') as f:
    pickle.dump([y_test],f)










