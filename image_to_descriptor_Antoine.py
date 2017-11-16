# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 10:22:44 2017

@author: Amira AYADI
"""

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path_area = "./created_data/resized_images/AREA/"
path_cubic = "./created_data/resized_images/CUBIC/"
nb_vid = 371
h = 80
w = 60
nb_img = 5
#images = np.zeros((nb_img, h*w*3))

'''Descriptor image'''

def img_descri(path=path_area):
    images = np.zeros((nb_vid,14400))
    target = np.zeros(nb_vid)
    categorie_rep = os.listdir(path_area)
    i=0
    categorie_rep.remove('.DS_Store')
    for cat in categorie_rep:
        cat_rep = os.listdir(path_area+cat)
        y = 0
        for img_rep in cat_rep:
            if img_rep!= '.DS_Store':
                y+=1
                img_liste=os.listdir(path_area+cat+"/"+img_rep)
                i+=1
                z= 0
                d= []
                for img in img_liste:
                    
                    if img!= '.DS_Store' and z<1:
                        lien = path_area+cat+"/"+img_rep+"/"+img
                        image = cv2.imread(lien)
                        image_flat = list(image.flatten())
                        d = d + image_flat
                        z+=1
                images[i-1] = d
                target[i-1] = int(cat)
        print(cat+' : ' + str(y))
                
                        
    return(images,target)

image,target=img_descri()


''' Classes très déséquilibrées
1001 : 8
1009 : 46
1011 : 73
1013 : 64
1014 : 135
1017 : 45
'''
'''Descriptor  color mean'''

def color_descri(path=path_area):
    images = np.zeros((nb_img,3))
    target = np.zeros(nb_img)
    categorie_rep = os.listdir(path_area)
    i=0
    for cat in categorie_rep:
        cat_rep = os.listdir(path_area+cat)
        for img_rep in cat_rep:
            img_liste=os.listdir(path_area+cat+"/"+img_rep)
            for img in img_liste:
                lien = path_area+cat+"/"+img_rep+"/"+img
                image = cv2.imread(lien)
                means = cv2.mean(image)
                means = means[:3]
                images[i] = means
                target[i] = int(cat)
                i+=1
    return(images,target)

image,target=color_descri()


'''Descriptor  color mean and standard deviation '''

def color_st_descri(path=path_area):
    images = np.zeros((nb_img,6))
    target = np.zeros(nb_img)
    categorie_rep = os.listdir(path_area)
    i=0
    for cat in categorie_rep:
        cat_rep = os.listdir(path_area+cat)
        for img_rep in cat_rep:
            img_liste=os.listdir(path_area+cat+"/"+img_rep)
            for img in img_liste:
                lien = path_area+cat+"/"+img_rep+"/"+img
                image = cv2.imread(lien)
                (means, stds) = cv2.meanStdDev(image)
                stats = np.concatenate([means, stds]).flatten()
                images[i] = stats
                target[i] = int(cat)
                i+=1
    return(images,target)

image,target=color_st_descri()


'''Descriptor 3D color Histogram '''

def td_hist_descri(path=path_area):
    images = np.zeros((nb_img,512))
    target = np.zeros(nb_img)
    categorie_rep = os.listdir(path_area)
    i=0
    for cat in categorie_rep:
        cat_rep = os.listdir(path_area+cat)
        for img_rep in cat_rep:
            img_liste=os.listdir(path_area+cat+"/"+img_rep)
            for img in img_liste:
                lien = path_area+cat+"/"+img_rep+"/"+img
                image = cv2.imread(lien)
                hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
                cv2.normalize(hist, hist)
                hist = hist.flatten()
                images[i] = hist
                target[i] = int(cat)
                i+=1
    return(images,target)

image,target=td_hist_descri()


import sklearn.model_selection as s



def evaluate(classifier, X, y): # (X,y) is a testing set
    return(mean(classifier.predict(X) != y))
    
class DataSet:
    def __init__(self,image,target):
            X = image
            y = target
            X_train, X_test, y_train, y_test = s.train_test_split(X, y, test_size = 0.2, random_state = 0)
            self.X_train = X_train
            self.X_test = X_test
            self.y_train = y_train
            self.y_test = y_test


    
                
d = DataSet(image,target)                
                
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(d.X_train, d.y_train) 

sol = neigh.predict(d.X_test)
neigh.score(d.X_test, d.y_test)

np.mean(sol!= d.y_test)

