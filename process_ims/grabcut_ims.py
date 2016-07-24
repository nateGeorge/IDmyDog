# takes images that have rectangles drawn
# applies grabcut
# saves background and foreground for further processing

from __future__ import print_function
import pandas as pd
import pickle as pk
import cv2
import os
import re
import numpy as np

def get_rect(rectI):
    # takes unsorted rectangle from pandas DF
    # returns rectangle for grabcut (x, y, dx, dy)
    rect = (min(bd[0][0], bd[1][0]), min(bd[0][1], bd[1][1]), abs(bd[1][0] - bd[0][0]), abs(bd[1][1] - bd[0][1]))
    return rect

BLUE = [255,0,0]        # rectangle color
RED = [0,0,255]         # PR BG
GREEN = [0,255,0]       # PR FG
BLACK = [0,0,0]         # sure BG
WHITE = [255,255,255]   # sure FG

DRAW_BG = {'color' : BLACK, 'val' : 0}
DRAW_FG = {'color' : WHITE, 'val' : 1}
DRAW_PR_FG = {'color' : GREEN, 'val' : 3}
DRAW_PR_BG = {'color' : RED, 'val' : 2}

mainImPath = '/media/nate/Windows/github/IDmyDog/scrape-ims/images/'

bb = pk.load(open('pickle_files/pDogs-bounding-boxes-clean.pd.pk', 'rb'))
bb.dropna(inplace=True)

startIm = 141 # got interrupted here

for i in range(startIm, bb.shape[0]):
    entry = bb.iloc[i]
    cropDir = mainImPath + entry.breed + '/cropped/bodies/'
    # make directories for foreground and background if needed
    for dd in ['fg', 'bg']:
        if not os.path.isdir(cropDir + dd):
            os.makedirs(cropDir + dd)
    # get filename of image
    imName = entry.path.split('/')[-1]
    sb = re.search('St. Bernard', imName)
    ext = re.search('\.\w', imName)
    print(imName)
    if ext:
        if sb:
            imName = ' '.join(imName.split('.')[0:2])
        else:
            imName = imName.split('.')[0]
    elif imName[-1]=='.':
        imName = imName[:-1]
    print(imName)
    image = cv2.imread(entry.path)
    orig = image.copy()
    bods = entry.bodies
    # only working now for single rectangle
    if len(bods) == 1:
        bd = bods[0]
        
        # do first iterations of grabcut
        bgdmodel = np.zeros((1,65),np.float64)
        fgdmodel = np.zeros((1,65),np.float64)
        rect = get_rect(bd)
        mask = np.zeros(image.shape[:2], dtype = np.uint8) # mask initialized to BG
        mask, bgdModel, fgdModel = cv2.grabCut(image, mask, rect, bgdmodel, fgdmodel, 5, cv2.GC_INIT_WITH_RECT)
        
        # second iterations of grabcut
        try:
            mask, bgdModel, fgdModel = cv2.grabCut(orig, mask, rect, bgdmodel, fgdmodel, 5, cv2.GC_INIT_WITH_MASK)
        except cv2.error as err:
            print(err)
        
        # make images with alpha channel
        b_channel, g_channel, r_channel = cv2.split(orig)
        # if probably background (2) or background (0), set to transparent (0)
        # otherwise set to fully opaque (255)
        a_channel = np.where((mask==2)|(mask==0), 0, 255).astype('uint8')
        foreground = cv2.merge((b_channel, g_channel, r_channel, a_channel))
        # same idea for background, just inverted
        a_channel_bg = np.where((mask==2)|(mask==0), 255, 0).astype('uint8')
        background = cv2.merge((b_channel, g_channel, r_channel, a_channel_bg))
        #cv2.imshow('fg', foreground)
        #cv2.imshow('bg', background)
        cv2.imwrite(cropDir + 'fg/' + imName + '.png', foreground)
        cv2.imwrite(cropDir + 'bg/' + imName + '.png', background)