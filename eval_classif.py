import numpy as np


from sklearn import cross_validation

def GetRankByCat(Y_test, Y_pred, print_table = False):
    """
    Param : (Y_test, Y_pred)
    Return : (precision, rappel, F_Mesure)
    """
    result = {}
    genre_unique = list(set(Y_test))
    n_test = len(Y_test)
    for g in genre_unique:
        result[g] = np.zeros((2,2))
    for i in range(len(Y_test)):
        cr = Y_test[i]
        cp = Y_pred[i]
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

    precision = {}
    rappel = {}
    F_Mesure = {}
    for key, val in result.items():
        result[key][1,1] = len(Y_test) - val.sum()
        if print_table:
            print(key)
            print(result[key])
        precision[key] = result[key][0, 0]/(result[key][0, 0]+result[key][0, 1])
        rappel[key] = result[key][0, 0]/(result[key][0, 0]+result[key][1, 0])
        F_Mesure[key] = (2*precision[key]*rappel[key])/(precision[key]+rappel[key])

    return(precision, rappel, F_Mesure)