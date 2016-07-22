# removes any bounding boxes that are really small (accidental clicks)

import cv2
import os
import pickle as pk
import numpy as np

bb = pk.load(open('pickle_files/pDogs-bounding-boxes.pd.pk', 'rb'))
bb.dropna(inplace=True)
toDrop = []


for i in range(bb.shape[0]):
    # check if any bounding boxes are small (accidental click)
    # remove them from the temp array and then rewrite the data
    # if it changed
    bods = bb.iloc[i].bodies
    newBods = bods.copy()
    changed = False
    for body in bods:
        bbDiffs = sum(abs(np.array(body[0])-np.array(body[1])))
        if bbDiffs < 20:
            print('removing', body, 'from index', i)
            changed = True
            newBods.remove(body)
    if changed:
        bb.iloc[i].bodies = newBods
    
    # do the same for the heads bounding boxes
    heads = bb.iloc[i].heads
    newHeads = heads.copy()
    changed = False
    for head in heads:
        bbDiffs = sum(abs(np.array(head[0])-np.array(head[1])))
        if bbDiffs < 20:
            print('removing', head, 'from index', i)
            changed = True
            newHeads.remove(head)
    if changed:
        bb.iloc[i].heads = newHeads
    # drop row if both bodies and heads bounding boxes are empty
    if newBods == [] and newHeads == []:
        toDrop.append(i)

bb.drop(bb.index[toDrop], inplace=True)
pk.dump(bb, open('pickle_files/pDogs-bounding-boxes-clean.pd.pk', 'wb'))