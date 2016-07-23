# takes dog images and pandas dataframe
# cuts out images of heads and bodies
# using bounding boxes from pandas DF
from __future__ import print_function
import pandas as pd
import pickle as pk
import cv2
import os
import imutils
import re

mainImPath = '/media/nate/Windows/github/IDmyDog/scrape-ims/images/'

bb = pk.load(open('pickle_files/pDogs-bounding-boxes-clean.pd.pk', 'rb'))
bb.dropna(inplace=True)
for i in range(bb.shape[0]):
    # make dir for cropped ims if doesn't already exist
    cropDir = mainImPath + bb.iloc[i].breed + '/cropped/'
    if not os.path.isdir(cropDir):
        os.makedirs(cropDir)
        for bp in ['heads','bodies']:
            if not os.path.isdir(cropDir + bp):
                os.makedirs(cropDir + bp)
    # crop out body from image
    bods = bb.iloc[i].bodies
    imPath = bb.iloc[i].path
    imName = bb.iloc[i].path.split('/')[-1]
    sb = re.search('St. Bernard', imName)
    ext = re.search('\.\w', imName)
    print(imName)
    if ext:
        if sb:
            imName = ' '.join(imName.split('.')[0:2])
        else:
            imName = imName.split('.')[0]
    print(imName)
    image = cv2.imread(imPath)
    cv2.imshow('original', image)
    for body in bods:
        ys = sorted([body[0][1], body[1][1]])
        xs = sorted([body[0][0], body[1][0]])
        crBod = image[ys[0]:ys[1], xs[0]:xs[1]]
        #cv2.imshow('body', crBod)
        cv2.imwrite(cropDir + 'bodies/' + imName + '.jpg', crBod)
    # crop out heads
    heads = bb.iloc[i].heads
    for head in heads:
        ys = sorted([head[0][1], head[1][1]])
        xs = sorted([head[0][0], head[1][0]])
        crBod = image[ys[0]:ys[1], xs[0]:xs[1]]
        #cv2.imshow('head', crBod)
        cv2.imwrite(cropDir + 'heads/' + imName + '.jpg', crBod)