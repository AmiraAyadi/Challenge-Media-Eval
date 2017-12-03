
'''
############################################################
#                       apprentissage knn                  #
############################################################
'''

#Importation des données
with open('./created_data/data/train.pkl','rb') as f:
        X_train,y_train = pickle.load(f)
with open('./created_data/data/X_test.pkl','rb') as f:
        X_test,y_test = pickle.load(f)

# nombre de voisin pour mean,sd = 15. concat hori = 6
from sklearn.neighbors import KNeighborsClassifier
cls = KNeighborsClassifier(n_neighbors=15)
cls.fit(hist_matrice,hist_y) 

sol = cls.predict(hist_matrice_test)
cls.score(hist_matrice_test, hist_y_test)
#40%

# nombre de voisin pour mean,sd = 15. concat hori = 6
from sklearn.neighbors import KNeighborsClassifier
cls = KNeighborsClassifier(n_neighbors=6)
cls.fit(lbp_matrice,lbp_y) 

sol = cls.predict(lbp_matrice_test)
cls.score(lbp_matrice_test, lbp_y_test)
#36%

from sklearn.neighbors import KNeighborsClassifier
cls = KNeighborsClassifier(n_neighbors=6)
cls.fit(c_matrice,c_y) 
sol = cls.predict(c_matrice_test)
cls.score(c_matrice_test, c_y_test)
#25%

# nombre de voisin pour mean,sd = 15. concat hori = 6 (avec 2 : hist et lbp)
from sklearn.neighbors import KNeighborsClassifier
cls = KNeighborsClassifier(n_neighbors=6)
cls.fit(X_train,y_train) 
sol = cls.predict(X_test)
cls.score(X_test, y_test)
#48%

# nombre de voisin pour mean,sd = 15. concat hori = 6 (avec 2 : hist et lbp et shape)
from sklearn.neighbors import KNeighborsClassifier
cls = KNeighborsClassifier(n_neighbors=6)
cls.fit(X_train,y_train) 
sol = cls.predict(X_test)
cls.score(X_test, y_test)
#48%

np.mean(sol!= y_test)
#meilleur score pouor l'instat 45,7% 
