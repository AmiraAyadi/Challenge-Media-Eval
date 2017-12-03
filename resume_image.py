# -*-coding:Utf-8 -*

import numpy as np
import cv2
import skimage
import matplotlib.pyplot as plt
import os
from math import *

def DistImg(img1, img2):
    return(sqrt(np.sum((img1-img2)**2)))

def ImageDiff(vid_rep, nb_image = 15):
    list_file_img = os.listdir(vid_rep)
    list_file_img_back = np.asarray(list_file_img)

    dist = np.zeros((len(list_file_img), len(list_file_img)))
    nb_file = len(list_file_img)

    a = 0
    for i in range(nb_file):
        img1_file = list_file_img[i]
        img1 = cv2.imread(vid_rep+"/"+img1_file,0)
        for j in range(len(list_file_img)):
            img2_file = list_file_img[j]
            img2 = cv2.imread(vid_rep+"/"+img2_file,0)
            dist[i,a+j] = DistImg(img1, img2)

    val_a_sup = np.argwhere(dist<=200)
    val_sup = {}
    for val in val_a_sup:
        if val[0] != val[1]:
            try:
                val_sup[val[0]].append(val[1])
            except:
                val_sup[val[0]] = [val[1]]

    print(val_sup)

    j = 0

    val_to_keep = []

    while True:
        print(j)
        if not val_sup:
            break
        key1 = val_sup[np.asscalar(sorted(val_sup.keys())[0])]
        del val_sup[np.asscalar(sorted(val_sup.keys())[0])]
        val_to_keep.extend(key1)
        if len(key1) > 1 :
            for a in key1:
                a = np.asscalar(a)
                dist[a, :] = 0
                dist[:, a] = 0 
            try:
                del val_sup[a]
            except:
                print("Clé pas supprimer")
                
        else:
            dist[key1, :] = 0
            dist[:, key1] = 0   
            try:
                del val_sup[key1[0]]
            except:
                print("Clé pas supprimer")
        j+=1 
    if dist[0, 1] == 0 and dist[1,2] != 0:
        imgs = [1]
    else : 
        imgs=[0]

    for i in range(nb_image):
        imgs.append(np.argmax(dist[imgs[-1], :]))
        dist[imgs[-2],:] = 0
        dist[:,imgs[-2]] = 0
    return(list_file_img_back[imgs])


if __name__ == '__main__':
    file_r = "created_data/resized_images/CUBIC/"
    file_w = "created_data/image_resume/"
    list_genre = os.listdir(file_r)
    j = 0
    for genre in list_genre:
        if not os.path.exists(file_w+genre):
            os.makedirs(file_w+genre)
        file_genre = file_r + genre
        film_list = os.listdir(file_genre)
        for film in film_list:
            print(file_w+genre+"/"+film)
            if not os.path.exists(file_w+genre+"/"+film):
                os.makedirs(file_w+genre+"/"+film)
            film_path = file_genre+"/"+film
            
            if len(os.listdir(film_path)) ==1:
                imgs = os.listdir(film_path)
                print("loool CESt TROO NULLE MDR XD")

            else:
                imgs = ImageDiff(film_path)
            print(imgs)
            for img in imgs:
                img_path = film_path+"/"+img
                img_img = cv2.imread(img_path)
                plt.imsave(file_w+genre+"/"+film+"/"+img, img_img)





