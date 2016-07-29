# compares the properties of the background and foreground of images
# foreground is the dog, background is everything else
# fg/bg were separated with cv2.grabCut()

from __future__ import print_function
import pandas as pd
import pickle as pk
import cv2
import os
import re
import json
import progressbar
import imutils
import numpy as np
import matplotlib.pyplot as plt
from mahotas.features import haralick
import threading
from multiprocessing import Pool
plt.style.use('seaborn-dark')

def make_fg_bg_hist_plot(fg, bg):
    # make a plot comparing color histograms of foreground to background
    f, axarr = plt.subplots(2, 2)
    r, g, b, a = cv2.split(fg)
    bData = np.extract(a>0, b)
    gData = np.extract(a>0, g)
    rData = np.extract(a>0, r)
    axarr[0,0].set_title("Foreground")
    axarr[0,0].set_ylabel("Normalized # of pixels")
    for chan, col in zip([rData, gData, bData], ['red', 'green', 'blue']):
        hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
        hist /= hist.sum() # normalize to compare images of different sizes
        axarr[0,0].plot(hist, color = col)
        axarr[0,0].set_xlim([0, 256])

    r, g, b, a = cv2.split(bg)
    bData = np.extract(a>0, b)
    gData = np.extract(a>0, g)
    rData = np.extract(a>0, r)
    axarr[0,1].set_title("Background")
    for chan, col in zip([rData, gData, bData], ['red', 'green', 'blue']):
        hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
        hist /= hist.sum() # normalize to compare images of different sizes
        axarr[0,1].plot(hist, color = col)
        axarr[0,1].set_xlim([0, 256])
    axarr[1,0].imshow(cv2.cvtColor(fg, cv2.COLOR_BGRA2RGBA))
    axarr[1,1].imshow(cv2.cvtColor(bg, cv2.COLOR_BGRA2RGBA))
    plt.show()

def get_fg_bg_color_hists(fg, bg):
    # returns normalized histograms of color for 
    r, g, b, a = cv2.split(fg)
    bData = np.extract(a>0, b)
    gData = np.extract(a>0, g)
    rData = np.extract(a>0, r)
    fgHist = {}
    for chan, col in zip([rData, gData, bData], ['red', 'green', 'blue']):
        fgHist[col] = cv2.calcHist([chan], [0], None, [256], [0, 256])
        fgHist[col] /= fgHist[col].sum() # normalize to compare images of different sizes

    r, g, b, a = cv2.split(bg)
    bData = np.extract(a>0, b)
    gData = np.extract(a>0, g)
    rData = np.extract(a>0, r)
    bgHist = {}
    for chan, col in zip([rData, gData, bData], ['red', 'green', 'blue']):
        bgHist[col] = cv2.calcHist([chan], [0], None, [256], [0, 256])
        bgHist[col] /= bgHist[col].sum() # normalize to compare images of different sizes
    
    return fgHist, bgHist

def get_fg_color_hists(fg):
    # returns normalized histograms of color for 
    r, g, b, a = cv2.split(fg)
    bData = np.extract(a>0, b)
    gData = np.extract(a>0, g)
    rData = np.extract(a>0, r)
    fgHist = {}
    for chan, col in zip([rData, gData, bData], ['red', 'green', 'blue']):
        fgHist[col] = cv2.calcHist([chan], [0], None, [256], [0, 256])
        fgHist[col] /= fgHist[col].sum() # normalize to compare images of different sizes
    
    return fgHist

def get_fg_bg_rects(fg):
    b, g, r, a = cv2.split(fg)
    h, w = fg.shape[:2]
    h -= 1
    w -= 1 # to avoid indexing problems
    rectDims = [10, 10] # w, h of rectangles
    hRects = h / rectDims[0]
    wRects = w / rectDims[1]
    fgRects = []
    bgRects = []
    for i in range(wRects):
        for j in range(hRects):
            pt1 = (i * rectDims[0], j * rectDims[1])
            pt2 = ((i + 1) * rectDims[0], (j + 1) * rectDims[1])
            # alpha is 0 over the part of the dog
            if a[pt1[1], pt1[0]] == 255 and a[pt2[1], pt2[0]] == 255:
                fgRects.append([pt1, pt2])
                #cv2.rectangle(fgcp, pt1, pt2, [0, 0, 255], 2)
            elif a[pt1[1], pt1[0]] == 0 and a[pt2[1], pt2[0]] == 0:
                bgRects.append([pt1, pt2])
                #cv2.rectangle(bgcp, pt1, pt2, [0, 0, 255], 2)
    
    return fgRects, bgRects

def get_fg_rects(fg):
    b, g, r, a = cv2.split(fg)
    h, w = fg.shape[:2]
    h -= 1
    w -= 1 # to avoid indexing problems
    rectDims = [10, 10] # w, h of rectangles
    hRects = h / rectDims[0]
    wRects = w / rectDims[1]
    fgRects = []
    for i in range(wRects):
        for j in range(hRects):
            pt1 = (i * rectDims[0], j * rectDims[1])
            pt2 = ((i + 1) * rectDims[0], (j + 1) * rectDims[1])
            # alpha is 0 over the part of the dog
            if a[pt1[1], pt1[0]] == 255 and a[pt2[1], pt2[0]] == 255:
                fgRects.append([pt1, pt2])
    
    return fgRects

def get_avg_hara(im, rects):
    # returns the haralick texture averaged over all rectangles in an image
    if len(rects)==0:
        return None
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    hara = 0
    for r in rects:
        # slice images as: img[y0:y1, x0:x1]
        hara += haralick(im[r[0][1]:r[1][1], r[0][0]:r[1][0]])
    hara /= (len(rects))
    return hara

def do_analysis(breed):
    global histNtext, rowCnt, pbar
    cropDir = mainImPath + breed + '/cropped/bodies/'
    fgDir = cropDir + 'fg/'
    fgFiles = os.listdir(fgDir)

    for fi in fgFiles:
        print(fi)
        print('')
        try:
            fg = cv2.imread(fgDir + fi, -1) # -1 tells it to load alpha channel
        except: # some weren't reading properly
            continue
        if fg!=None:
            if fg.shape[1] > 450:
                fg = imutils.resize(fg, width = 450)
            fgRects = get_fg_rects(fg)
            fgHara = get_avg_hara(fg, fgRects)
            fgHist = get_fg_color_hists(fg)
            if None in [fgRects, fgHara]:
                continue
            histNtext.loc[rowCnt] = [breed, fi, fgHara]
            rowCnt += 1
            try:
                pbar.update(rowCnt)
            except:
                pass
    
    pk.dump(histNtext, open(pDir + 'fgHara-full-13x4.pd.pk', 'wb'))

# load configuration
with open('../config.json', 'rb') as f:
    config = json.load(f)

mainImPath = config['image_dir']
pDir = config['pickle_dir']

bb = pk.load(open(pDir + 'pDogs-bounding-boxes-clean.pd.pk', 'rb'))
bb.dropna(inplace=True)

widgets = ["Calculating Haralick features: ", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
pbar = progressbar.ProgressBar(maxval=1480, widgets=widgets).start() # think maxval should be bb.shape[0]*6.3, but know its 1459 post-mortem

rowCnt = 0
histNtext = pd.DataFrame(columns=['breed', 'file', 'fgHaralick'])

p = Pool(8)
p.map(do_analysis, list(sorted(bb.breed.unique().tolist())))

'''
for breed in list(sorted(bb.breed.unique().tolist())):
    cropDir = mainImPath + breed + '/cropped/bodies/'
    fgDir = cropDir + 'fg/'
    fgFiles = os.listdir(fgDir)

    for fi in fgFiles:
        print(fi)
        print('')
        try:
            fg = cv2.imread(fgDir + fi, -1) # -1 tells it to load alpha channel
        except: # some weren't reading properly
            continue
        if fg!=None:
            if fg.shape[1] > 450:
                fg = imutils.resize(fg, width = 450)
            fgRects = get_fg_rects(fg)
            fgHara = get_avg_hara(fg, fgRects)
            fgHist = get_fg_color_hists(fg)
            if None in [fgRects, fgHara]:
                continue
            histNtext.loc[rowCnt] = [breed, fi, fgHara]
            rowCnt += 1
            try:
                pbar.update(rowCnt)
            except:
                pass
    
    pk.dump(histNtext, open('../pickle_files/fgHara-full-13x4-new.pd.pk', 'wb'))
'''