from __future__ import print_function
import pandas as pd
import pickle as pk
import cv2
import os
import re
import numpy as np
import mahotas.features as mh
import matplotlib.pyplot as plt
plt.style.use('seaborn-dark')

mainImPath = '/media/nate/Windows/github/IDmyDog/scrape-ims/images/'

bb = pk.load(open('pickle_files/pDogs-bounding-boxes-clean.pd.pk', 'rb'))
bb.dropna(inplace=True)

for breed in sorted(bb.breed.unique().tolist()):
    cropDir = mainImPath + breed + '/cropped/bodies/'
    fgDir = cropDir + 'fg/'
    bgDir = cropDir + 'bg/'
    fgFiles = os.listdir(fgDir)
    
    for fi in fgFiles:
        print(fi)
        fg = cv2.imread(fgDir + fi, -1)
        bg = cv2.imread(bgDir + fi, -1) # -1 tells it to load alpha channel
        if fg!=None and bg!=None:
            fgChans = {}
            fgChans['b','g','r','a'] = cv2.split(fg)
            haralickFG = mh.haralick(fg).mean(0)
            haralickBG = mh.haralick(bg).mean(0)
            f, ax = plt.subplots(figsize=(7, 7))