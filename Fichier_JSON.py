# -*- coding: utf-8 -*-
#############################################################
#                Création du fichier JSON                   #
#############################################################

import os
import re
import json
import numpy as np
import pandas as pd
from lxml import etree
from itertools import combinations


files_meta = os.listdir("Dataset/metadata/")
files_shot = os.listdir("Dataset/shots/XML_shot")
files_trans = os.listdir("Dataset/trans/")

data = {}

for i,(meta,shot,trans) in enumerate(zip(files_meta,files_shot,files_trans)):
    
    data[i] = {}
    
    # Métadonnées
    
    metadata = {}
            
    tree = etree.parse("Dataset/metadata/"+ meta)
    root = tree.getroot()
    
    metadata["title"] = root.find('title').text
    metadata["description"] = root.find('description').text
    metadata["explicit"] = root.find('explicit').text
    metadata["duration"] = root.find('duration').text
    metadata["url"] = root.find('url').text
    metadata['license'] = {}
    for license_tag in root.find('license'):
        metadata['license'][license_tag.tag] = license_tag.text
    tag = []
    for V_tag in root.find('tags'):
        tag.append(V_tag.text)
    metadata['tags'] = tag

    metadata['uploader'] = {}
    for uploader_child in root.find('uploader'):
        metadata['uploader'][uploader_child.tag] = uploader_child.text

    metadata['file'] = {}
    for file_child in root.find('file'):
        metadata['file'][file_child.tag] = file_child.text
        
    data[i]["metadata"] = metadata
    
    # Shots
    
    shots_prof = {}
    try :
        tree = etree.parse("Dataset/shots/XML_shot/" + shot)
        root = tree.getroot()
        shots_prof["Segment"] = {}
        for segs in root.find("Segments"):
            for seg in segs.getiterator():
                if seg.tag =="Segment":
                    shots_prof["Segment"] = dict(seg.attrib)
                if seg.tag == "index":
                    shots_prof["Segment"]["index"] = seg.text
                if seg.tag == "KeyFrameID":
                    shots_prof["Segment"]["KeyFrame"] = {}
                    shots_prof["Segment"]["KeyFrame"]["time"] = dict(seg.attrib)
                    shots_prof["Segment"]["KeyFrame"]["name"] = seg.text
    except etree.XMLSyntaxError:
        shots_prof[i] = "empty"
        pass
    
    data[i]["shots"] = shots_prof
    
    # Trans
    
    transcription = {}
    all_words = []
    tree = etree.parse("Dataset/trans/" + trans)
    root = tree.getroot()
    for chanelist in root.find("ChannelList"):
        for chanel in chanelist.getiterator():
            transcription["Channel"] = dict(chanel.attrib)
    transcription["Speaker"] = {}
    for speaklist in root.find("SpeakerList"):
        for j,speaker in enumerate(speaklist.getiterator()):
            transcription["Speaker"]["speaker_"+str(j)] = dict(speaker.attrib)
    words = []
    transcription["SpeechSegment"] = {}
    for seg_list in root.find("SegmentList"):
        for sp_seg in seg_list.getiterator():
            if sp_seg.tag=="SpeechSegment":
                transcription["SpeechSegment"]["meta"] = dict(sp_seg.attrib)
            if sp_seg.tag=="Word":
                transcription["SpeechSegment"][sp_seg.text] = dict(sp_seg.attrib)
                words.append(sp_seg.text)
        all_words.extend(words)
        transcription["SpeechSegment"]["Seg_text"] = words
    transcription["text"] = all_words
    

    #Suppression des mots dont le score est le plus bas  (a optimiser)
    d = transcription["SpeechSegment"]
    for word1, word2 in combinations(d.keys(), r = 2):
        if word1 not in ["meta","Seg_text"] and word2 not in ["meta","Seg_text"]:
            if d[word1]["stime"]==d[word2]["stime"]:
                if d[word1]["conf"] > d[word2]["conf"]:
                    if word2 in d["Seg_text"]:
                        d["Seg_text"].remove(word2)
                    if word2 in all_words:
                        all_words.remove(word2)
                else:
                    if word1 in d["Seg_text"]:
                        d["Seg_text"].remove(word1)
                    if word1 in all_words:
                        all_words.remove(word1)
    data[i]["Trans"] = transcription

    # Ajout de la catégorie
    
    genre_arr = pd.read_csv("./created_data/Genre.csv", sep=";", header= None)
    for vid in data.keys():
        for c,t in zip(genre_arr[0],genre_arr[1]):
            if re.findall(str(data[vid]["metadata"]["file"]["filename"]),str(t)):
                data[vid]["categorie"] = np.asscalar(np.int16(c))
    
with open("./data_file.json", "w") as m_file:
    json.dump(data, m_file)

