# compares the properties of the background and foreground of images
# foreground is the dog, background is everything else
# fg/bg were separated with cv2.grabCut()

from __future__ import print_function
import pandas as pd
import pickle as pk
import cv2
import os
import re
import progressbar
import numpy as np
import matplotlib.pyplot as plt
from mahotas.features import haralick
plt.style.use('seaborn-dark')

mainImPath = '/media/nate/Windows/github/IDmyDog/scrape-ims/images/'

bb = pk.load(open('pickle_files/pDogs-bounding-boxes-clean.pd.pk', 'rb'))
bb.dropna(inplace=True)

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

def get_avg_hara(im, rects):
    # returns the haralick texture averaged over all rectangles in an image
    hara = 0
    for r in rects:
        # slice images as: img[y0:y1, x0:x1]
        hara += haralick(im[r[0][1]:r[1][1], r[0][0]:r[1][0]]).mean(0)
    hara /= (len(rects) + 1)
    return hara

widgets = ["Comparing: ", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
pbar = progressbar.ProgressBar(maxval=bb.shape[0], widgets=widgets).start()

rowCnt = 0
histNtext = pd.DataFrame(columns=['file', 'fgHaralick', 'bgHaralick', 'fgHist', 'bgHist'])
for i, breed in enumerate(sorted(bb.breed.unique().tolist())):
    cropDir = mainImPath + breed + '/cropped/bodies/'
    fgDir = cropDir + 'fg/'
    bgDir = cropDir + 'bg/'
    fgFiles = os.listdir(fgDir)
    
    for fi in fgFiles:
        print(fi)
        fg = cv2.imread(fgDir + fi, -1)
        bg = cv2.imread(bgDir + fi, -1) # -1 tells it to load alpha channel
        if fg!=None and bg!=None:
            #make_fg_bg_hist_plot(fg, bg)
            fgRects, bgRects = get_fg_bg_rects(fg)
            fgHara = get_avg_hara(fg, fgRects)
            bgHara = get_avg_hara(bg, bgRects)
            fgHist, bgHist = get_fg_bg_color_hists(fg, bg)
            histNtext.loc[rowCnt] = [fi, fgHara, bgHara, fgHist, bgHist]
            rowCnt += 1
    pickle.dump(histNtext, open('pickle_files/histNtext.pd.pk', 'wb'))
    pbar.update(i)
            
            
        
pbar.finish()