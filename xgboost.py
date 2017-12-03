# -*- coding: utf-8 -*-
import pickle
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score


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
X_train = np.squeeze(X_train, axis=1)


X_test = np.array(X_test)
X_test = np.squeeze(X_test, axis=0)
X_test = np.squeeze(X_test, axis=1)

y_train = np.squeeze(y_train, axis=0)
y_test = np.squeeze(y_test, axis=0)

model = XGBClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))