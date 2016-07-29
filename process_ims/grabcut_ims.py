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
import json
import progressbar

def get_rect(rectI):
    # takes unsorted rectangle from pandas DF
    # returns rectangle for grabcut (x, y, dx, dy)
    rect = (min(bd[0][0], bd[1][0]), min(bd[0][1], bd[1][1]), abs(bd[1][0] - bd[0][0]), abs(bd[1][1] - bd[0][1]))
    return rect

# load configuration
with open('../config.json', 'rb') as f:
    config = json.load(f)

mainImPath = config['image_dir']
pDir = config['pickle_dir']

BLUE = [255,0,0]        # rectangle color
RED = [0,0,255]         # PR BG
GREEN = [0,255,0]       # PR FG
BLACK = [0,0,0]         # sure BG
WHITE = [255,255,255]   # sure FG

DRAW_BG = {'color' : BLACK, 'val' : 0}
DRAW_FG = {'color' : WHITE, 'val' : 1}
DRAW_PR_FG = {'color' : GREEN, 'val' : 3}
DRAW_PR_BG = {'color' : RED, 'val' : 2}

bb = pk.load(open(pDir + 'pDogs-bounding-boxes-clean.pd.pk', 'rb'))
bb.dropna(inplace=True)

startIm = 0
#startIm = 141 # got interrupted here...uncomment if you need to resume mid-dataframe

widgets = ["grabCutting images: ", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
pbar = progressbar.ProgressBar(maxval=bb.shape[0]-startIm, widgets=widgets).start()

for i in range(startIm, bb.shape[0]):
    try:
        pbar.update(i)
    except:
        pass
    entry = bb.iloc[i]
    gcDir = mainImPath + entry.breed + '/grabcut/'
    # make fg and bg directory if needed
    if not os.path.isdir(gcDir):
            os.makedirs(gcDir)
    for dd in ['fg/', 'bg/']:
        if not os.path.isdir(gcDir + dd):
            os.makedirs(gcDir + dd)
    
    # get filename of image
    imName = entry.path.split('/')[-1]
    sb = re.search('St. Bernard', imName)
    ext = re.search('\.\w', imName)
    #print(imName) # for debugging
    if ext:
        if sb:
            imName = ' '.join(imName.split('.')[0:2])
        else:
            imName = imName.split('.')[0]
    
    elif imName[-1]=='.':
        imName = imName[:-1]
    
    #print(imName) # for debugging
    try:
        image = cv2.imread(entry.path)
    except:
        continue
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
        cv2.grabCut(image, mask, rect, bgdmodel, fgdmodel, 5, cv2.GC_INIT_WITH_RECT)
        
        # second iterations of grabcut - sometimes it doesn't work
        try:
            cv2.grabCut(orig, mask, rect, bgdmodel, fgdmodel, 5, cv2.GC_INIT_WITH_MASK)
        except cv2.error as err:
            print(err)
        
        # make images with alpha channel
        b_channel, g_channel, r_channel = cv2.split(orig)
        # if probably background (2) or background (0), set to trasparent (0)
        # otherwise make opaque (255)
        a_channel = np.where((mask==2)|(mask==0), 0, 255).astype('uint8')
        foreground = cv2.merge((b_channel, g_channel, r_channel, a_channel))
        # same idea for background, just inverted
        a_channel_bg = np.where((mask==2)|(mask==0), 255, 0).astype('uint8')
        background = cv2.merge((b_channel, g_channel, r_channel, a_channel_bg))
        #cv2.imshow('fg', foreground) # for debugging
        #cv2.imshow('bg', background)
        cv2.imwrite(gcDir + 'fg/' + imName + '.png', foreground)
        cv2.imwrite(gcDir + 'bg/' + imName + '.png', background)

pbar.finish()