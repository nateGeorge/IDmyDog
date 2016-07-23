from __future__ import print_function
import dlib
from skimage import io

import pandas as pd
import pickle as pk
import cv2
import os
import imutils
import re
import numpy as np
import pylab as plt

#global image, mask, rect

BLUE = [255,0,0]        # rectangle color

mainImPath = '/media/nate/Windows/github/IDmyDog/scrape-ims/images/'

bb = pk.load(open('pickle_files/pDogs-bounding-boxes-clean.pd.pk', 'rb'))
bb.dropna(inplace=True)

for i in range(bb.shape[0]):
    entry = bb.iloc[i]
    image = cv2.imread(entry.path)
    bods = entry.bodies
    if len(bods) == 1:
        mask = np.zeros(image.shape[:2], dtype = np.uint8) # mask initialized to PR_BG
        bd = bods[0]
        print(bd)
        rect = (bd[1][0], bd[0][1], -bd[1][0] + bd[0][0], bd[1][1] - bd[0][1])
        print(rect)
        #cv2.rectangle(image, (bd[1][0], bd[0][1]), (bd[0][0], bd[1][1]), BLUE, 2)
        
        bgdmodel = np.zeros((1,65),np.float64)
        fgdmodel = np.zeros((1,65),np.float64)
        
        cv2.grabCut(image,mask,rect,bgdmodel,fgdmodel,5,cv2.GC_INIT_WITH_RECT)
        cv2.grabCut(image,mask,rect,bgdmodel,fgdmodel,5,cv2.GC_INIT_WITH_MASK)
        
        #mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
        #image = image*mask2[:,:,np.newaxis]
        #plt.imshow(image)
        #plt.show()
        
        while True:
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
        exit()