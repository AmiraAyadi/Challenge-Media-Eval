#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 15:11:53 2017

@author: AntoineP
"""
from Challenge_Media_Eval_master.texte import ClassifTfidf
from keras.models import load_model
from Challenge_Media_Eval_master.RNN import getTrainTest
import numpy as np


X_train,X_test,y_train, y_test = getTrainTest()

model2 = load_model('ModeleRNN3.h5')
scoreV = 69
scoreT = 84
coefT = scoreT/(scoreV+scoreT)
coefV = scoreV/(scoreV+scoreT)

#Probas Images
V = model2.predict_proba(X_test)
#Probas Texte
T = ClassifTfidf()


probaCombine = coefV*V +coefT*T

#Calcul du score
np.mean(probaCombine.argmax(1) == y_test.argmax(1))

    




















