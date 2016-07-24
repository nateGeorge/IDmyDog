from __future__ import print_function
import pandas as pd
import pickle as pk
import cv2
import os
import numpy as np
import pylab as plt

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

for i in range(bb.shape[0]):
    entry = bb.iloc[i]
    image = cv2.imread(entry.path)
    orig = image.copy()
    bods = entry.bodies
    if len(bods) == 1:
        bd = bods[0]
        #cv2.rectangle(image, (bd[1][0], bd[0][1]), (bd[0][0], bd[1][1]), BLUE, 2)
        
        bgdmodel = np.zeros((1,65),np.float64)
        fgdmodel = np.zeros((1,65),np.float64)
        
        rect = get_rect(bd)
        mask = np.zeros(image.shape[:2], dtype = np.uint8) # mask initialized to BG
        mask, bgdModel, fgdModel = cv2.grabCut(image, mask, rect, bgdmodel, fgdmodel, 5, cv2.GC_INIT_WITH_RECT)
        
        mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
        image = orig*mask2[:,:,np.newaxis]
        #cv2.imshow('first iter', image)
        mask, bgdModel, fgdModel = cv2.grabCut(orig, mask, rect, bgdmodel, fgdmodel, 5, cv2.GC_INIT_WITH_MASK)
        
        mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
        image = orig*mask2[:,:,np.newaxis]
        b_channel, g_channel, r_channel = cv2.split(orig)
        a_channel = np.where((mask==2)|(mask==0), 0, 255).astype('uint8')
        foreground = cv2.merge((b_channel, g_channel, r_channel, a_channel))
        a_channel_bg = np.where((mask==2)|(mask==0), 255, 0).astype('uint8')
        background = cv2.merge((b_channel, g_channel, r_channel, a_channel_bg))
        cv2.imshow('alpha', foreground)
        cv2.imwrite('foreground.png', foreground)
        cv2.imwrite('background.png', background)
        cv2.waitKey(0)

        
        #cv2.imshow('2nd iter', image)
        #cv2.waitKey(0)
        
        '''while True:
            cv2.imshow('image', image)
            mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
            img = image*mask2[:,:,np.newaxis]
            cv2.imshow('', img)
            mask2 = np.where((mask==1) + (mask==3),255,0).astype('uint8')
            output = cv2.bitwise_and(image, image, mask=mask2)
            cv2.imshow('output', output)
            
            k = 0xFF & cv2.waitKey(1)

            # key bindings
            if k == 27:         # esc to exit
                break
            if k == ord('n'):
                print("runnin")
                bgdmodel = np.zeros((1,65),np.float64)
                fgdmodel = np.zeros((1,65),np.float64)
                cv2.grabCut(image,mask,rect,bgdmodel,fgdmodel,1,cv2.GC_INIT_WITH_MASK)
            '''