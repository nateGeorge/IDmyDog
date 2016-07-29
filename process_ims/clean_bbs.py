# removes any bounding boxes that are really small (accidental clicks)
from __future__ import print_function
import cv2
import os
import pickle as pk
import numpy as np
import json

# load configuration
with open('../config.json', 'rb') as f:
    config = json.load(f)

mainImPath = config['image_dir']
pDir = config['pickle_dir']

bb = pk.load(open(pDir + 'pDogs-bounding-boxes.pd.pk', 'rb'))
bb.dropna(inplace=True)
toDrop = []

for i in range(bb.shape[0]):
    # check if any bounding boxes are small (accidental click)
    # remove them from the temp array and then rewrite the data
    # if it changed
    imPath = bb.iloc[i].path
    image = cv2.imread(imPath)
    h, w = image.shape[:2]
    bods = bb.iloc[i].bodies
    newBods = bods.copy()
    changed = False
    bodyCnt = 0
    for body in bods:
        y1 = min([body[0][1], body[1][1]])
        y2 = max([body[0][1], body[1][1]])
        x1 = min([body[0][0], body[1][0]])
        x2 = max([body[0][0], body[1][0]])
        bc = [[x1, y1], [x2, y2]]
        changedBC = False
        bbDiffs = sum(abs(np.array(body[0])-np.array(body[1])))
        if bbDiffs < 20:
            print('removing', body, 'from index', i)
            changed = True
            newBods.remove(body)
        else:
            # take care of case where box is outside of image
            # first check xs
            if bc[0][0] < 0:
                bc[0][0] = 0
                changedBC = True
            if bc[1][0] > w:
                bc[1][0] = w
                changedBC = True
            # then ys
            if bc[0][1] < 0:
                bc[0][0] = 0
                changedBC = True
            if bc[1][1] > h:
                bc[1][1] = h
                changedBC = True
        if changedBC:
            newBods[bodyCnt] = bc
            print('changed:')
            print(body)
            print(bc)
        bodyCnt += 1
    bb.iloc[i].bodies = newBods
    
    # do the same for the heads bounding boxes
    heads = bb.iloc[i].heads
    newHeads = heads.copy()
    changed = False
    headCnt = 0
    for head in heads:
        y1 = min([head[0][1], head[1][1]])
        y2 = max([head[0][1], head[1][1]])
        x1 = min([head[0][0], head[1][0]])
        x2 = max([head[0][0], head[1][0]])
        bc = [[x1, y1], [x2, y2]]
        changedBC = False
        bbDiffs = sum(abs(np.array(head[0])-np.array(head[1])))
        if bbDiffs < 20:
            print('removing', head, 'from index', i)
            changed = True
            newHeads.remove(head)
        else:
            # take care of case where box is outside of image
            # xs first
            if bc[0][0] < 0:
                bc[0][0] = 0
                changedBC = True
            if bc[1][0] > w:
                bc[1][0] = w
                changedBC = True
            # ys next
            if bc[0][1] < 0:
                bc[0][0] = 0
                changedBC = True
            if bc[1][1] > h:
                bc[1][1] = h
                changedBC = True
        if changedBC:
            newHeads[headCnt] = bc
            changed = True
        headCnt += 1
    bb.iloc[i].heads = newHeads
    # drop row if both bodies and heads bounding boxes are empty
    if newBods == [] and newHeads == []:
        toDrop.append(i)

bb.drop(bb.index[toDrop], inplace=True)
pk.dump(bb, open(pDir + 'pDogs-bounding-boxes-clean.pd.pk', 'wb'), protocol=2)