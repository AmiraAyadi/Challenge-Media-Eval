# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
import pandas as pd
from shutil import copytree
from random import randint
import matplotlib.pyplot as plt
import sklearn.model_selection as s
import pickle
import Image


#verification
path = "./created_data/videos - Copie/"
categorie_rep = os.listdir(path)
for cat in categorie_rep:
   cat_rep = os.listdir(path+'/'+cat)
   for rep in cat_rep:
       vid_img = os.listdir(path+'/'+cat+'/'+rep)
       if len(vid_img) != 15 :
           print("C:/Users/Amira AYADI/Desktop/M2 SID/Challenge Media Eval/created_data/videos/"+cat+'/'+rep+"\n")
       
	

# division du train/test
os.makedirs("./created_data/data")
os.makedirs("./created_data/data/train")
os.makedirs("./created_data/data/test")
liste_video=[]
liste_cat = []
x_test_l=[]
x_train_l=[]
y_test=[]
y_train=[]
categorie_rep = os.listdir(path)
for cat in categorie_rep:
    cat_rep = os.listdir(path+cat)
    for rep in cat_rep:
        liste_video.append(rep)
        liste_cat.append(cat)
random=[]
for i in range(121):
    random.append(randint(0, 605))
for vid in liste_video:
    lien = path+liste_cat[liste_video.index(vid)]+'/'+vid
    if liste_video.index(vid) in random:
        copytree(lien, "./created_data/data/test/"+vid)
        x_test_l.extend(list(np.repeat(lien,15)))
        y_test.extend(list(np.repeat(liste_cat[liste_video.index(vid)],15)))
    else:
        copytree(lien, "./created_data/data/train/"+vid)
        x_train_l.extend(list(np.repeat(lien,15)))
        y_train.extend(list(np.repeat(liste_cat[liste_video.index(vid)],15)))

path_train= "./created_data/data/train"
path_test="./created_data/data/test"

hist_matrice,hist_y=Image.td_hist_descri(path_train,"train")
hist_matrice_test,hist_y_test=Image.td_hist_descri(path_test,"test")

cm_matrice,cm_y=Image.color_descri(path_train,"train")
cm_matrice_test,cm_y_test=Image.color_descri(path_test,"test")


texture_matrice,texture_y=Image.texture_descri(path_train,"train")
texture_matrice_test,texture_y_test=Image.texture_descri(path_test,"test")


shape_matrice,shape_y=Image.shape_descri(path_train,"train")
shape_matrice_test,shape_y_test=Image.shape_descri(path_test,"test")

zer_matrice,zer_y = Image.zer_descri(path_train,"train")
zer_matrice_test,zer_y_test= Image.zer_descri(path_test,"test")


lbp_matrice,lbp_y = Image.lbp_descri(path_train,"train")
lbp_matrice_test,lbp_y_test=Image.lbp_descri(path_test,"test")

c_matrice,c_y = Image.contour(path_train,"train")
c_matrice_test,c_y_test=Image.contour(path_test,"test")


descripteur = np.concatenate((hist_matrice,lbp_matrice,shape_matrice), axis=1)
descripteur_test = np.concatenate((hist_matrice_test, lbp_matrice_test,shape_matrice_test), axis = 1)

# mean_sd
X_train,y_train = Image.mean_sd(descripteur,"train")
X_test,y_test = Image.mean_sd(descripteur_test,"test")
y_train = np.asarray(y_train)
y_test = np.asarray(y_test)

# mean_sd
X_train,y_train = Image.mean_sd(shape_matrice,"train")
X_test,y_test = Image.mean_sd(shape_matrice_test,"test")
y_train = np.asarray(y_train)
y_test = np.asarray(y_test)


# concat hori
X_train,y_train = Image.concat_hori(descripteur,"train")
X_test,y_test = Image.concat_hori(descripteur_test,"test")
y_train = np.asarray(y_train)
y_test = np.asarray(y_test)

# Enregistrement de x et y train 
with open('./created_data/data/train.pkl', 'wb') as f:
    pickle.dump([x_train_l, y_train], f)

# Enregistrement de x et y test 
with open('./created_data/data/test.pkl', 'wb') as f:
    pickle.dump([x_test_l, y_test], f)

        
