# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 15:26:52 2017

@author: Amira AYADI
"""
# Image

import os
import cv2
import pickle
import numpy as np
import pandas as pd
import mahotas as mt



'''
Création des descripteurs : 3D COLOR, Texture, contour
'''
# 3D COLOR HIST
def td_hist_descri(path,t): #t = train ou test
    with open('./created_data/data/'+t+'.pkl','rb') as f:
        x,y = pickle.load(f)
    nb_img = len(x)
    matrice = np.zeros((nb_img,512))
    index=0
    rep=os.listdir(path)
    for img_rep in rep:
        img_liste=os.listdir(path+"/"+img_rep)
        for img in img_liste:
            lien = path+"/"+img_rep+"/"+img
            image = cv2.imread(lien)
            hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            cv2.normalize(hist, hist)
            hist = hist.flatten()
            matrice[index] = hist
            index+=1
    return(matrice,y)

# Color 

def color_descri(path,t):
    with open('./created_data/data/'+t+'.pkl','rb') as f:
        x,y = pickle.load(f)
    nb_img = len(x)
    matrice = np.zeros((nb_img,3))
    index=0
    rep=os.listdir(path)
    for img_rep in rep:
        img_liste=os.listdir(path+"/"+img_rep)
        for img in img_liste:
            lien = path+"/"+img_rep+"/"+img
            image = cv2.imread(lien)
            means = cv2.mean(image)
            means = means[:3]
            matrice[index] = means
            index+=1
    return(matrice,y)

# Texture tp
t="train"
def texture_descri(path,t):
    with open('./created_data/data/'+t+'.pkl','rb') as f:
        x,y_ = pickle.load(f)
    nb_img = len(x)
    matrice = np.zeros((nb_img,24))
    index=0
    rep = os.listdir(path)
    for img_rep in rep:
        img_liste=os.listdir(path+"/"+img_rep)
        for img in img_liste:
            lien = path+"/"+img_rep+"/"+img
            descripteur = []
            image = cv2.imread(lien)
            image = cv2.resize(image, (100,200))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
            image = np.log(np.abs(np.fft.fft2(image)))
            t = pd.DataFrame(image)
            t = t[0:int(len(t)/2)] #récupère uniquement la moitié des lignes
            t = np.array(t)
            pasX = 3
            pasY = 6
            dimX = np.shape(t)[0]
            dimY = np.shape(t)[1]
            X = np.arange(0,dimX,int(dimX/pasX))
            Y = np.arange(0,dimY-int(dimY/pasY),int(dimY/pasY))
            for x in X:  
                for y in Y:    
                    if (np.mean(t[x:x+int(dimX/pasX),y:y+int(dimY/pasY)] **2)) !=0:
                        bloc = np.log(np.mean(t[x:x+int(dimX/pasX),y:y+int(dimY/pasY)] **2))
                    else:
                        bloc=0
                    descripteur.append(bloc)
            matrice[index] = descripteur
            index+=1
    return(matrice,y_)

# SHAPE HU MOMENT
def shape_descri(path,t): #t = train ou test
    with open('./created_data/data/'+t+'.pkl','rb') as f:
        x,y = pickle.load(f)
    nb_img = len(x)
    matrice = np.zeros((nb_img,7))
    index=0
    rep=os.listdir(path)
    for img_rep in rep:
        img_liste=os.listdir(path+"/"+img_rep)
        for img in img_liste:
            lien = path+"/"+img_rep+"/"+img
            image = cv2.imread(lien)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            shape = cv2.HuMoments(cv2.moments(image)).flatten()
            # pour supp bruit et skewness
            #shape = np.sign(shape) * np.log10(np.abs(shape))
            matrice[index] = shape
            index+=1
    return(matrice,y)

# Zarnike shape 

class ZernikeMoments:
	def __init__(self, radius):
		# store the size of the radius that will be
		# used when computing moments
		self.radius = radius
 
	def describe(self, image):
		# return the Zernike moments for the image
		return mt.features.zernike_moments(image, self.radius)

desc = ZernikeMoments(21)

def zer_descri(path,t): #t = train ou test
    with open('./created_data/data/'+t+'.pkl','rb') as f:
        x,y = pickle.load(f)
    nb_img = len(x)
    matrice = np.zeros((nb_img,25))
    index=0
    rep=os.listdir(path)
    for img_rep in rep:
        img_liste=os.listdir(path+"/"+img_rep)
        for img in img_liste:
            lien = path+"/"+img_rep+"/"+img
            image = cv2.imread(lien)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.copyMakeBorder(image, 15, 15, 15, 15,cv2.BORDER_CONSTANT, value = 255)
            thresh = cv2.bitwise_not(image)
            thresh[thresh > 0] = 255
            outline = np.zeros(image.shape, dtype = "uint8")
            _,cnts,_ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)            
            try:
                cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[0]
                cv2.drawContours(outline, [cnts], -1, 255, -1)
            except:
                moments = desc.describe(outline)
            moments = desc.describe(outline)
            matrice[index] = moments
            index+=1
    return(matrice,y)

#Local Binary Patterns

# import the necessary packages
from skimage import feature
 
class LocalBinaryPatterns:
	def __init__(self, numPoints, radius):
		# store the number of points and radius
		self.numPoints = numPoints
		self.radius = radius
 
	def describe(self, image, eps=1e-7):
		# compute the Local Binary Pattern representation
		# of the image, and then use the LBP representation
		# to build the histogram of patterns
		lbp = feature.local_binary_pattern(image, self.numPoints,
			self.radius, method="uniform")
		(hist, _) = np.histogram(lbp.ravel(),
			bins=np.arange(0, self.numPoints + 3),
			range=(0, self.numPoints + 2))
 
		# normalize the histogram
		hist = hist.astype("float")
		hist /= (hist.sum() + eps)
 
		# return the histogram of Local Binary Patterns
		return hist
desc = LocalBinaryPatterns(24, 8)

def lbp_descri(path,t): #t = train ou test
    with open('./created_data/data/'+t+'.pkl','rb') as f:
        x,y = pickle.load(f)
    nb_img = len(x)
    matrice = np.zeros((nb_img,26))
    index=0
    rep=os.listdir(path)
    for img_rep in rep:
        img_liste=os.listdir(path+"/"+img_rep)
        for img in img_liste:
            lien = path+"/"+img_rep+"/"+img
            image = cv2.imread(lien)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            hist = desc.describe(gray)
            matrice[index] = hist
            index+=1
    return(matrice,y)

def contour(path,t):
    with open('./created_data/data/'+t+'.pkl','rb') as f:
        x,y = pickle.load(f)
    nb_img = len(x)
    matrice = np.zeros((nb_img,100*200))
    index=0
    rep=os.listdir(path)
    for img_rep in rep:
        img_liste=os.listdir(path+"/"+img_rep)
        for img in img_liste:
            lien = path+"/"+img_rep+"/"+img
            image = cv2.imread(lien)
            image = cv2.resize(image, (100,200))
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray = cv2.bilateralFilter(gray, 11, 17, 17)
            edged = cv2.Canny(gray, 30, 200)
            edged = edged.flatten()
            matrice[index] = edged
            index+=1
    return(matrice,y)

from pyefd import elliptic_fourier_descriptors
def efd_feature(contour):
    coeffs = elliptic_fourier_descriptors(contour, order=10, normalize=True)
    return coeffs.flatten()[3:]


'''
Modélisation : Moyenne et variance, concaténation horizental , concaténation vertical
'''
def mean_sd(matrice,t):
    with open('./created_data/data/'+t+'.pkl','rb') as f:
        x,y = pickle.load(f)
    new_matrice = np.zeros((int(len(x)/15),matrice.shape[1]*2))
    targ = np.zeros((int(len(x)/15)))
    index=0
    for ligne in range(0,len(x),15):
        une_video = matrice[ligne:ligne+15]
        mean = une_video.mean(axis=0)
        mean = np.reshape(mean, (1,matrice.shape[1]))
        sd = une_video.std(axis=0)
        sd = np.reshape(sd,(1,matrice.shape[1]))
        new_ligne = np.concatenate((mean,sd),axis=1)
        new_matrice[index]=new_ligne
        targ[index] = y[ligne]
        index+=1
    return (new_matrice,targ)

def concat_hori(matrice,t):
    with open('./created_data/data/'+t+'.pkl','rb') as f:
        x,y = pickle.load(f)
    une_video = matrice[0:15]
    new_matrice = np.zeros((int(len(x)/15),une_video.shape[1]*une_video.shape[0]))
    targ = np.zeros((int(len(x)/15)))
    index=0
    for ligne in range(0,len(x),15):
        une_video = matrice[ligne:ligne+15]
        new_ligne = une_video.reshape((1,une_video.shape[1]*une_video.shape[0]))
        new_matrice[index]=new_ligne
        targ[index] = y[ligne]
        index+=1
    return (new_matrice,targ)
