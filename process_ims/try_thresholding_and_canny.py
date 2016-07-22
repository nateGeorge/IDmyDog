# takes dog images and pandas dataframe
# cuts out images of heads and bodies
# using bounding boxes from pandas DF
from __future__ import print_function
import pandas as pd
import pickle as pk
import cv2
import os
import imutils

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
    image = cv2.imread(bb.iloc[i].path)
    cv2.imshow('original', image)
    for body in bods:
        print(body)
        ys = sorted([body[0][1],body[1][1]])
        xs = sorted([body[0][0],body[1][0]])
        crBod = image[ys[0]:ys[1], xs[0]:xs[1]]
        cv2.imshow('cropped', crBod)
        can = imutils.auto_canny(crBod)
        cv2.imshow('canny', can)
        gray = cv2.cvtColor(crBod, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        (T, threshInv) = cv2.threshold(blurred, 0, 255,
	                    cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        cv2.imshow("Threshold", threshInv)
        cv2.waitKey(0)
        #exit()