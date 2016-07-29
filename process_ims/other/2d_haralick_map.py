from __future__ import print_function
import pandas as pd
import pickle as pk
import cv2
import os
import re
import progressbar
import imutils
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mahotas.features import haralick
import json
from sklearn.decomposition import PCA
plt.style.use('seaborn-dark')

def get_fg_bg_rects(fg):
    b, g, r, a = cv2.split(fg)
    h, w = fg.shape[:2]
    h -= 1
    w -= 1 # to avoid indexing problems
    rectDims = [10, 10] # h, w of rectangles
    hRects = h / rectDims[0]
    wRects = w / rectDims[1]
    fgRects = []
    bgRects = []
    for i in range(wRects):
        for j in range(hRects):
            pt1 = (i * rectDims[0], j * rectDims[1])
            pt2 = ((i + 1) * rectDims[0], (j + 1) * rectDims[1])
            # alpha is 255 over the part of the dog
            if a[pt1[1], pt1[0]] == 255 and a[pt2[1], pt2[0]] == 255:
                fgRects.append([pt1, pt2])
                #cv2.rectangle(fgcp, pt1, pt2, [0, 0, 255], 2) # for debugging
            elif a[pt1[1], pt1[0]] == 0 and a[pt2[1], pt2[0]] == 0:
                bgRects.append([pt1, pt2])
                #cv2.rectangle(bgcp, pt1, pt2, [0, 0, 255], 2)
    
    return fgRects, bgRects

def get_avg_hara(im, rects):
    # returns the haralick texture averaged over all rectangles in an image
    if len(rects)==0:
        return None
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    hara = 0
    for r in rects:
        # slice images as: img[y0:y1, x0:x1]
        hara += haralick(im[r[0][1]:r[1][1], r[0][0]:r[1][0]]).mean(0)
    hara /= (len(rects))
    return hara

def make_hara_map(im, rects):
    # draws heatmap of haralick texture PCA dim1 variance
    if len(rects)==0:
        return None
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    hara = []
    for r in rects:
        # slice images as: img[y0:y1, x0:x1]
        hara.append(pcaFG.transform(haralick(im[r[0][1]:r[1][1], r[0][0]:r[1][0]]).mean(0).reshape(1, -1)))
    
    hara = np.array(hara)
    haraMean = np.mean(hara, axis=0)
    haraStd = np.std(hara, axis=0)
    haraMins = np.min(hara, axis=0)
    haraMaxs = np.max(hara, axis=0)
    norm = (haraMaxs-haraMins)
    copy = im.copy()
    copy = cv2.cvtColor(copy, cv2.COLOR_BGRA2RGBA)
    im = cv2.cvtColor(im, cv2.COLOR_BGRA2RGBA)
    
    for i in range(hara.shape[0]):
        brightScale = 255*(hara[i] - haraMins)/norm
        bright = brightScale[0][0]
        r = rects[i]
        cv2.rectangle(copy, r[0], r[1], [0, bright, 0, 255], -1)
    
    f, axarr = plt.subplots(2, 1)
    axarr[0].imshow(copy)
    axarr[1].imshow(im)
    plt.show()


# load configuration
with open('../../config.json', 'rb') as f:
    config = json.load(f)

mainImPath = config['image_dir']
pDir = config['pickle_dir']

pcaFG = pk.load(open(pDir + 'pcaFG.pk', 'rb'))

bb = pk.load(open(pDir + 'pDogs-bounding-boxes-clean.pd.pk', 'rb'))
bb.dropna(inplace=True)

# do something like sorted(bb.breed.unique().tolist())[50:] to check another breed
for breed in sorted(bb.breed.unique().tolist())[50:]:
    print('breed:', breed)
    cropDir = mainImPath + breed + '/grabcut/'
    fgDir = cropDir + 'fg/'
    fgFiles = os.listdir(fgDir)
    
    for fi in fgFiles:
        try:
            fg = cv2.imread(fgDir + fi, -1) # -1 tells it to load alpha channel
        except Exception as err:
            print('exception:', err)
            continue
        fgRects, bgRects = get_fg_bg_rects(fg)
        make_hara_map(fg, fgRects)
        