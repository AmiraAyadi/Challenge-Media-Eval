import os
import numpy as np
import json
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC 
from sklearn.linear_model import SGDClassifier

from sklearn import cross_validation

def GetRankByCat(Y_test, Y_pred):
    """
    Take Y_test and the prediction by the classifier
    Return a dictionnary with a numpy matrix with real 
    in line and prediction in column
    I think we should find a measurement to return only a list
    """
    for i in range(len(Y_test)):
        cr = Y_test[i]
        cp = classe_predict[i]
        if cr == cp:
            try:
                result[cr][0,0] += 1
            except:
                result[cr][0,0] = 1
        if cr != cp:
            try:
                result[cr][1,0] += 1
                result[cp][0,1] += 1
            except:
                result[cr][1,0] = 1
                result[cp][0,1] = 1

    for key, val in result.items():
        result[key][1,1] = len(Y_test) - val.sum()

    return(result)
    

with open('data_file.json', 'r') as f:
    datas = json.load(f)

words_delete = stopwords.words('english')
ponctuation = [".", ",", ";", ":", "etc","{fw}"]
words_delete.extend(ponctuation)
all_words = []
y = np.zeros((len(datas)))
j = 0
for vid in datas.keys():
    v = datas[vid]["Trans"]["text"]
    v = [re.sub(r"^ ", "", w.rstrip()).lower() for w in v]
    v = [word for word in v if word not in words_delete]
    all_words.append(" ".join(v))
    y[j] = datas[vid]["categorie"]
    j += 1  

tfidf_sp = TfidfVectorizer(max_df=0.95, min_df=2, max_features=100000) 
tfidf_gl = TfidfVectorizer(max_df=0.95, min_df=2, max_features=100000) 

X_train, X_test, Y_train, Y_test = train_test_split(all_words, y, test_size = 0.2)

tfidf_tr = tfidf_sp.fit_transform(X_train)

neigh = KNeighborsClassifier(n_neighbors = 6)
neigh.fit(tfidf_tr, Y_train)

tfidf_te = tfidf_sp.transform(X_test)

print(neigh.score(tfidf_te, Y_test))

RF = RandomForestClassifier(200, max_depth=5)
RF.fit(tfidf_tr, Y_train)
print(RF.score(tfidf_te, Y_test))

SV = SVC()
SV.fit(tfidf_tr, Y_train)
print(SV.score(tfidf_te, Y_test))

clf = SGDClassifier(loss='log', penalty='l2',alpha=1e-3, n_iter=5, random_state=42) #Notre SVC classifier
clf.fit(tfidf_tr, Y_train) #On entraine le classifier
print(clf.score(tfidf_te, Y_test))

tfidf = tfidf_gl.fit_transform(all_words)
clf = SGDClassifier(loss='log', penalty='l2',alpha=1e-3, n_iter=5, random_state=42) #Notre SVC classifier
clf.fit(tfidf, y) #On entraine le classifier
scores = cross_validation.cross_val_score(clf, tfidf, y, cv=5)
print(scores)

clf = SGDClassifier(loss='log', penalty='l2',alpha=1e-3, n_iter=5, random_state=42)
clf.fit(tfidf_tr, Y_train)

result = {}
genre_unique = list(set(Y_test))
n_test = len(Y_test)
for g in genre_unique:
    result[g] = np.zeros((2,2))

classe_predict = clf.predict(tfidf_te)

for key, val in GetRankByCat(Y_test, classe_predict).items():
    print(key)
    print(val)





