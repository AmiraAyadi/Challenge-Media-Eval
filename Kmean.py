# -*- coding: utf-8 -*-

from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd 

#Importation des données 
with open('./created_data/data/X_train.pkl','rb') as f:
        X_train = pickle.load(f)
with open('./created_data/data/X_test.pkl','rb') as f:
        X_test = pickle.load(f)
with open('./created_data/data/y_train.pkl','rb') as f:
        y_train = pickle.load(f)
with open('./created_data/data/y_test.pkl','rb') as f:
        y_test = pickle.load(f)

X = np.concatenate((X_train,X_test), axis=0)
y = np.concatenate((y_train,y_test), axis=0)
y = np.array(y).astype('int64')


# Fitting K-Means to the dataset

kmeans = KMeans(n_clusters = 8, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X)
y = pd.DataFrame(y,columns=["y"])
y_kmeans = pd.DataFrame(y_kmeans,columns=["y_kmeans"])
y_a = pd.concat([y_kmeans,y], axis=1, join_axes=[y.index])
y_a = y_a.astype(int)

matrice_purete = y_a.groupby(["y_kmeans","y"]).agg("size")
d = pd.DataFrame(matrice_purete)

# Now I can count the labels for each cluster..
count0 = list(y_kmeans).count(0)
count1 = list(y_kmeans).count(1)
count2 = list(y_kmeans).count(2)
count3 = list(y_kmeans).count(3)
count4 = list(y_kmeans).count(4)
count5 = list(y_kmeans).count(5)
count6 = list(y_kmeans).count(6)
count7 = list(y_kmeans).count(7)

