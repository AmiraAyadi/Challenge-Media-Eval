# -*- coding: utf-8 -*-
import os
import cv2
from matplotlib import pyplot as plt


# Lecture des données

def Resize_img():
        
    size = (100, 100)
    path_w1 = "./created_data/resized_images/AREA/"
    path_w2 = "./created_data/resized_images/CUBIC/"
    path_r = "./created_data/videos_images/"
    categorie_rep = os.listdir(path_r)

    for cat in categorie_rep:
        video_rep = os.listdir(path_r+cat)
        if not os.path.exists(path_w1+cat):
            os.makedirs(path_w1+cat)
        if not os.path.exists(path_w2+cat):
            os.makedirs(path_w2+cat)
        
        for vid_im in video_rep:
            vid_img = os.listdir(path_r+cat+"/"+vid_im)
            if not os.path.exists(path_w1+cat+"/"+vid_im):
                os.makedirs(path_w1+cat+"/"+vid_im)
            if not os.path.exists(path_w2+cat+"/"+vid_im):
                os.makedirs(path_w2+cat+"/"+vid_im)
            lien = path_r+cat+"/"+vid_im
            del vid_img[-1]
            for nom_image in vid_img:
                image = lien + "/" + nom_image
                print(image)
                try:
                    image_v = cv2.imread(image)

                    plt.imsave(path_w1+cat+"/"+vid_im+ "/" + nom_image, cv2.resize(image_v, size, interpolation = cv2.INTER_AREA))
                    plt.imsave(path_w2+cat+"/"+vid_im+"/" +nom_image, cv2.resize(image_v, size, interpolation = cv2.INTER_AREA))


                    #plt.imshow(image)
                except:
                    print(image)
                    os.remove(image)


Resize_img()

    
    
