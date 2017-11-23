#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 18:20:10 2017

@author: AntoineP
"""
from keras.layers.recurrent import LSTM
from keras.optimizers import Adam, RMSprop
import pickle
import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation,Flatten

#On récupère les features créées par Xception


def getTrainTest():
    with open('./created_data2/data/X_train.pkl','rb') as f:
            X_train = pickle.load(f)
    with open('./created_data2/data/X_test.pkl','rb') as f:
            X_test = pickle.load(f)
    with open('./created_data2/data/y_train.pkl','rb') as f:
            y_train = pickle.load(f)
    with open('./created_data2/data/y_test.pkl','rb') as f:
            y_test = pickle.load(f)
            
    X_train = np.array(X_train)
    X_train = np.squeeze(X_train, axis=0)
    
    
    
    X_test = np.array(X_test)
    X_test = np.squeeze(X_test, axis=0)
    
    
    y_train = np.squeeze(y_train, axis=0)
    y_test = np.squeeze(y_test, axis=0)
    
    
    
    #def lstm():
    """Build a simple LSTM network. We pass the extracted features from
    our CNN to this model predomenently."""
    
    X_train = np.reshape(X_train,(int(6015/15),15,2048))
    X_test = np.reshape(X_test,(int(2940/15),15,2048))
    
    yTr = []
    for i in range(len(y_train)):
        if i%15==0 :
            yTr.append(y_train[i])
    y_train = np.array(yTr)
    
    yTe = []
    for i in range(len(y_test)):
        if i%15==0 :
            yTe.append(y_test[i])
    y_test = np.array(yTe)
    
    y_train = np.array(y_train).astype('int64')
    y_train[y_train ==1001] = 0
    y_train[y_train ==1009] = 1
    y_train[y_train ==1011] = 2
    y_train[y_train ==1013] = 3
    y_train[y_train ==1014] = 4
    y_train[y_train ==1016] = 5
    y_train[y_train ==1017] = 6
    y_train[y_train ==1019] = 7
    
    
    y_test = np.array(y_test).astype('int64')
    y_test[y_test ==1001] = 0
    y_test[y_test ==1009] = 1
    y_test[y_test ==1011] = 2
    y_test[y_test ==1013] = 3
    y_test[y_test ==1014] = 4
    y_test[y_test ==1016] = 5
    y_test[y_test ==1017] = 6
    y_test[y_test ==1019] = 7
    
    
    #On "dummifie" les catégories
    nb_classes = 8
    y_train = np_utils.to_categorical(y_train, nb_classes)
    y_test = np_utils.to_categorical(y_test, nb_classes)
    return X_train,X_test,y_train,y_test




if __name__ == '__main__':
    # Model.
    model = Sequential()
    model.add(LSTM(2048, return_sequences=False,
                   input_shape=(15,2048),
                   dropout=0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(8, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=1e-5, decay=1e-6),metrics=['accuracy'])    
    
    model.summary()
    
    batch_size = 20
    epochs=100
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs,  verbose=1, validation_data=(X_test, y_test))
    
    loss,acc = model.evaluate(X_test, y_test,  verbose=0)
    print('The accuracy on the test set is ',(acc*100),'%')
    
    model.save('./ModeleRNN3.h5')
    
    from keras.models import load_model
    
    model2 = load_model('ModeleRNN3.h5')










