# loads dog images and bounding boxes around heads and bodies
# honestly this would take way too long to go through by hand since there are ~1500 images
# this is intended as more of a spot check
from __future__ import print_function
import cv2
import os
import pickle as pk
import json

with open('../../config.json', 'rb') as f:
    config = json.load(f)

mainImPath = config['image_dir']
pDir = config['pickle_dir']

pDogs = pk.load(open(pDir + 'pDogs-bounding-boxes-clean.pd.pk', 'rb'))
bb = pDogs.dropna()

for i in range(bb.shape[0]):
    image = cv2.imread(mainImPath + bb.iloc[i].breed + '/' + bb.iloc[i].path.split('/')[-1])
    clone = image.copy()
    for body in bb.iloc[i].bodies:
        cv2.rectangle(clone, body[0], body[1], (0, 255, 0), 2)
    cv2.imshow('',clone)
    cv2.waitKey(0)
    
for i in range(bb.shape[0]):
    image = cv2.imread(mainImPath + bb.iloc[i].breed + '/' + bb.iloc[i].path.split('/')[-1])
    clone = image.copy()
    for head in bb.iloc[i].heads:
        cv2.rectangle(clone, head[0], head[1], (0, 255, 0), 2)
    cv2.imshow('',clone)
    cv2.waitKey(0)