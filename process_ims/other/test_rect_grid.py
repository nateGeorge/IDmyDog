# tests getting rectangles inside and outside of masked area
# mask is from alpha channel
import cv2
import pandas as pd
import pickle as pk
import os
import json

# load configuration
with open('../../config.json', 'rb') as f:
    config = json.load(f)

mainImPath = config['image_dir']
pDir = config['pickle_dir']

bb = pk.load(open(pDir + 'pDogs-bounding-boxes-clean.pd.pk', 'rb'))
bb.dropna(inplace=True)

for breed in sorted(bb.breed.unique().tolist()):
    cropDir = mainImPath + breed + '/grabcut/'
    fgDir = cropDir + 'fg/'
    bgDir = cropDir + 'bg/'
    fgFiles = os.listdir(fgDir)
    
    for fi in fgFiles:
        print(fi)
        if fi=='Affenpinscher1-2.png': # change filename here to check others
            fg = cv2.imread(fgDir + fi, -1)
            bg = cv2.imread(bgDir + fi, -1)
            break
    break

b, g, r, a = cv2.split(fg)
fgcp = fg.copy()
bgcp = bg.copy()
h, w = fg.shape[:2]
h -= 1
w -= 1 # to avoid indexing problems
rectDims = [10, 10] # w, h of rectangles
hRects = h / rectDims[0]
wRects = w / rectDims[1]
print(hRects, wRects)
fgRects = []
bgRects = []
for i in range(wRects):
    for j in range(hRects):
        pt1 = (i * rectDims[0], j * rectDims[1])
        pt2 = ((i + 1) * rectDims[0], (j + 1) * rectDims[1])
        print(pt1, pt2)
        # alpha is 0 over the part of the dog
        if a[pt1[1], pt1[0]] == 255 and a[pt2[1], pt2[0]] == 255:
            fgRects.append([pt1, pt2])
            cv2.rectangle(fgcp, pt1, pt2, [0, 0, 255], 2)
        elif a[pt1[1], pt1[0]] == 0 and a[pt2[1], pt2[0]] == 0:
            bgRects.append([pt1, pt2])
            cv2.rectangle(bgcp, pt1, pt2, [0, 0, 255], 2)

cv2.imshow('foreground', fgcp), cv2.imshow('background', bgcp), cv2.waitKey(0)