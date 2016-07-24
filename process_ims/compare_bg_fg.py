# compares the properties of the background and foreground of images
# foreground is the dog, background is everything else
# fg/bg were separated with cv2.grabCut()

from __future__ import print_function
import pandas as pd
import pickle as pk
import cv2
import os
import re
import numpy as np
import matplotlib.pyplot as plt
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
            make_fg_bg_hist_plot(fg, bg)
        
        '''fg_mask = np.where(a==255, 1, 0).astype('uint8')
        bg_mask = np.where(a==255, 0, 1).astype('uint8')
        bgr = cv2.merge((b, g, r))
        fg = bgr * fg_mask[:, :, np.newaxis]
        bg = bgr * bg_mask[:, :, np.newaxis]
        
        cv2.imshow('fg', fg)
        cv2.imshow('bg', bg)
        cv2.waitKey(0)'''