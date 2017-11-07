from lxml import etree
import os
import pandas as pd
import numpy as np

title = []
description = []
explicit = []
duration = []
url = []
license = []
tags = []
uploader = []
file = []



files = os.listdir("metadata/")

tree = etree.parse("metadata/"+files[0])
root = tree.getroot()


node_name = []
for child in root:
    node_name.append(child.tag)

print(node_name)

data = np.empty((6, len(node_name)), dtype = np.dtype.str)

for i in range(6):
    
    data[i,0] = root.find('title').text
    data[i,1] = root.find('description').text
    data[i,2] = root.find('explicit').text
    data[i,3] = root.find('duration').text
    data[i,4] = root.find('url').text
        

    data[i,5] = {}
    for license_tag in root.find('license'):
        data[i,5][license_tag.tag] = license_tag.text

    tags.append([])
    for V_tag in root.find('tags'):
        tags[-1].append(V_tag.text)


    uploader.append({})
    for uploader_child in root.find('uploader'):
        uploader[-1][uploader_child.tag] = uploader_child.text

    file.append({})
    for file_child in root.find('file'):
        file[-1][file_child.tag] = file_child.text


print(data)



