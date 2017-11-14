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
nb_img = 12041
h = 80
w = 60
#images = np.zeros((nb_img, h*w*3))

'''Descriptor image'''

def img_descri(path=path_area):
    images = np.zeros((nb_img,h*w*3))
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
                image_flat = image.flatten()
                images[i] = image_flat
                target[i] = int(cat)
                i+=1
    return(images,target)

image,target=img_descri()

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


    
                
                
                
        
        
