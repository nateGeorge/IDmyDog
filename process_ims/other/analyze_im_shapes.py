import cv2
import pandas as pd
import pickle as pk
import os
import json
import numpy as np

# load configuration
with open('../../config.json', 'rb') as f:
    config = json.load(f)

mainImPath = config['image_dir']
pDir = config['pickle_dir']

bb = pk.load(open(pDir + 'pDogs-bounding-boxes-clean.pd.pk', 'rb'))
bb.dropna(inplace=True)

Hs = []
Ws = []
dims = []
minH = np.inf
minW = np.inf
for i in range(bb.shape[0]):
    try:
        im = cv2.imread(mainImPath + bb.iloc[i].breed + '/' + bb.iloc[i].path.split('/')[-1])
        h, w, _ = im.shape
        Hs.append(h)
        Ws.append(w)
        dims.append([h, w])
        if h < minH:
            minH = h
        if w < minW:
            minW = w
    except:
        continue
