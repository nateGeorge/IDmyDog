import cv2
import pandas as pd
import pickle as pk
import os
import numpy as np

mainImPath = '/media/nate/Windows/github/IDmyDog/scrape-ims/images/'

bb = pk.load(open('pickle_files/pDogs-bounding-boxes-clean.pd.pk', 'rb'))
bb.dropna(inplace=True)

Hs = []
Ws = []
dims = []
minH = np.inf
minW = np.inf
for imPath in bb.path.tolist():
    try:
        im = cv2.imread(imPath)
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
