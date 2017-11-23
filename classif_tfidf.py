# -*-coding:Utf-8 -*

import pickle
import os
import re
import json

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import scipy

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier

from nltk.corpus import stopwords
from nltk.wsd import lesk
from nltk.corpus import wordnet as wn

from eval_classif import *

#### Recuperation lemma superieur ####
def DefHyper(word, floor = 3):
    """
    Takes : word, floor
    Return : word

    Il prend en entrer un mot et le niveau semantique à garder
    Puis il récupère le mot au niveau sémantique (ex: citron -> fruit)
    Et le retourne s'il le trouve
    """
    try:
        synset_word = wn.synsets(word)
        new_def = synset_word[0]
        depth_len = new_def.min_depth()
        nb_hyper = depth_len - floor
        for i in range(nb_hyper):
            new_def = new_def.hypernyms()[0]
        new_def = new_def.name()
    except:
        new_def = word
    return new_def

#### Recuperation Metadata ####
def Clean_Meta(data):
    """
    Takes : data from json (all metadata)
    Return : all_words (each line = video) , y  (category of videos in each line)
    
    Il prend en entrer les data du JSON.
    Puis pour chaque video, il recupère le texte dans les metadata, supprime les stopwords et la ponctuation
    Puis on récupère la catégorie de la vidéo
    """
    words_delete = stopwords.words('english')
    ponctuation = [".", ",", ";", ":", "etc","{fw}"]
    words_delete.extend(ponctuation)
    all_words = []
    y = np.zeros((len(data)))

    j = 0
    for vid in data.keys():
        # print(vid)
        words = []

        words_title = re.sub(r"[^A-Za-z ]", "", data[vid]["metadata"]["title"]).split()
        words_title = [re.sub(r"^ ", "", w.rstrip()).lower() for w in words_title]
        words_title = [word for word in words_title if word not in words_delete]
        words.append(" ".join(words_title).lower())

        words_desc = re.sub(r" +", " ", re.sub(r"[^A-Za-z ]", " ", data[vid]["metadata"]["description"])).split()
        words_desc = [re.sub(r"^ ", "", w.rstrip()).lower() for w in words_desc]
        words_desc = [word for word in words_desc if word not in words_delete]
        words.append(" ".join(words_desc))

        words_tags = data[vid]["metadata"]["tags"]
        words_tags_wordnet = [DefHyper(word) for word in words_tags]
        words_tags_wordnet = " ".join(words_tags_wordnet)
        words_tags = " ".join(words_tags)
        words.append(" ".join([words_tags, words_tags_wordnet]))

        all_words.append(" ".join(words))

        y[j] = data[vid]["categorie"]
        # if j >2:
        #     break
        j += 1  
    
    return(all_words, y)


def GetTfIdfMeta(change = False):
    """
    Return : Tfidf of metadatas, y_category
    """
    if os.path.exists('created_data/tfidf_meta') and not change:
        with open("created_data/tfidf_meta", 'rb') as fichier:
            mon_depickler = pickle.Unpickler(fichier)
            tfidf_gl_t = mon_depickler.load()
            y = mon_depickler.load()
        return(tfidf_gl_t, y)
    else:
        with open('data_file.json', 'r') as f:
            datas = json.load(f)
        all_words, y = Clean_Meta(datas)

        tfidf_gl = TfidfVectorizer(max_df=0.95, min_df=2, max_features=100000) 

        tfidf_gl_t = tfidf_gl.fit_transform(all_words)

        with open("created_data/tfidf_meta", 'wb') as fichier:
            mon_pickler = pickle.Pickler(fichier)
            mon_pickler.dump(tfidf_gl_t)
            mon_pickler.dump(y)

        return(tfidf_gl_t, y)


#### Recuperation Transcription ####
def Clean_Transc(data):
    """
    Takes : data from json (all transcription)
    Return : all_words (each line = video) , y  (category of videos in each line)
    
    Il prend en entrer les data du JSON.
    Puis pour chaque video, il recupère la transcritpion, supprime les stopwords et la ponctuation
    On appelle DefHyper qui met au même niveau semantique les mots et on l'ajoute au reste des mots
    Puis on récupère la catégorie de la vidéo
    """
    words_delete = stopwords.words('english')
    ponctuation = [".", ",", ";", ":", "etc","{fw}"]
    words_delete.extend(ponctuation)

    all_words = []
    y = np.zeros((len(data)))

    j = 0
    for video in data.keys():
        words = data[video]["Trans"]["text"]

        words = [re.sub(r"^ ", "", word.rstrip()).lower() for word in words]
        words = [word for word in words if word not in words_delete]

        wordnet_clean_word = [DefHyper(word) for word in words]

        sent = " ".join(words)
        wordnet_clean_sent = " ".join(wordnet_clean_word)

        all_words.append(wordnet_clean_sent + " "+sent)

        y[j] = data[video]["categorie"]
        j += 1 
    return(all_words, y)


def GetTfIdfTrans(change = False):
    """
    Return : Tfidf of Transcription, y_category
    """
    if os.path.exists('created_data/tfidf_transc') and not change:
        with open("created_data/tfidf_transc", 'rb') as fichier:
            mon_depickler = pickle.Unpickler(fichier)
            tfidf_gl_t = mon_depickler.load()
            y = mon_depickler.load()
        return(tfidf_gl_t, y)
    else:
        with open('data_file.json', 'r') as f:
            datas = json.load(f)
        
        all_words, y = Clean_Transc(datas)

        tfidf_gl = TfidfVectorizer(max_df=0.95, min_df=2, max_features=100000) 

        tfidf_gl_t = tfidf_gl.fit_transform(all_words)

        with open("created_data/tfidf_transc", 'wb') as fichier:
            mon_pickler = pickle.Pickler(fichier)
            mon_pickler.dump(tfidf_gl_t)
            mon_pickler.dump(y)

        return(tfidf_gl_t, y)
    

def ClassifTfidf():
    tfidf_meta, y_meta = GetTfIdfMeta()
    tfidf_trans, y_transc = GetTfIdfTrans()
    
    with open('data_file.json', 'r') as f:
        data = json.load(f)

    with open("created_data/data/train.pkl", "rb") as f:
        my_unpickler = pickle.Unpickler(f)
        train_path = my_unpickler.load()

    train_file = [j.split("/")[-1] for j in train_path[0]]


    with open("created_data/data/test.pkl", "rb") as f:
        my_unpickler = pickle.Unpickler(f)
        test_path = my_unpickler.load()
    
    test_file = [j.split("/")[-1] for j in train_path[0]]
    
    id_train = []
    id_test = []
    for video in data.keys():
        if data[video]["metadata"]["file"]["filename"]+".ogv" in train_file:
            id_train.append(video)
        else:
            id_test.append(video)
        
    

    tfidf = scipy.sparse.hstack((tfidf_meta, tfidf_trans)).tocsr()
    tfidf_train = tfidf[id_train]
    tfidf_test = tfidf[id_test]

    Y_train = [y_transc[int(i)] for i in id_train]
    Y_test = [y_transc[int(i)] for i in id_test]

    # X_train, X_test, Y_train, Y_test = train_test_split(tfidf, y_transc, test_size = 0.2)

    clf = SGDClassifier(loss='epsilon_insensitive', penalty='none',alpha=1e-3, n_iter=25, random_state=42)
    clf.fit(tfidf_train, Y_train)
    print(clf.score(tfidf_test, Y_test))
    return clf.predict(tfidf_test)
    
if __name__ == '__main__':
    ClassifTfidf()

