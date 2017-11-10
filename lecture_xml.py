# -*-coding:Utf-8 -*

from lxml import etree
import os
import pandas as pd
import numpy as np
import json

files = os.listdir("Dataset/metadata/")

tree = etree.parse("Dataset/metadata/"+files[0])
root = tree.getroot()

node_name = []
for child in root:
    node_name.append(child.tag)

data = {}

for i in range(len(files)):

    print(files[i])

    tree = etree.parse("Dataset/metadata/"+files[i])
    root = tree.getroot()
    data[i] = {}

    data[i]["title"] = root.find('title').text
    data[i]["description"] = root.find('description').text
    data[i]["explicit"] = root.find('explicit').text
    data[i]["duration"] = root.find('duration').text
    data[i]["url"] = root.find('url').text
        

    data[i]['license'] = {}
    for license_tag in root.find('license'):
        data[i]['license'][license_tag.tag] = license_tag.text

    tag = []
    for V_tag in root.find('tags'):
        tag.append(V_tag.text)
    data[i]['tags'] = tag

    data[i]['uploader'] = {}
    for uploader_child in root.find('uploader'):
        data[i]['uploader'][uploader_child.tag] = uploader_child.text

    data[i]['file'] = {}
    for file_child in root.find('file'):
        data[i]['file'][file_child.tag] = file_child.text


with open("created_data/metadata_file.json", "w") as m_file:
    json.dump(data, m_file)




