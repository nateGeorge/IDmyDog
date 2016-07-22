# loads dog images and bounding boxes around heads and bodies

import cv2
import os
import pickle as pk

pDogs = pk.load(open('pickle_files/pDogs-bounding-boxes.pd.pk', 'rb'))
bb = pDogs.dropna()

for i in bb.shape[0]:
    image = cv2.imread(bb.iloc[i].path)