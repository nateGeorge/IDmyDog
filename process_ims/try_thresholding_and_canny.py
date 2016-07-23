# takes dog images and pandas dataframe
# cuts out images of heads and bodies
# using bounding boxes from pandas DF
from __future__ import print_function
import pandas as pd
import pickle as pk
import cv2
import os
import imutils
from skimage.filters import threshold_adaptive
from skimage.morphology import disk
from skimage.filters import threshold_otsu, rank
from skimage.util import img_as_ubyte
import pylab as plt

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
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    # global thresholding--not a good idea
    # (T, threshInv) = cv2.threshold(blurred, 0, 255,
    #                cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    # local thresholding--much better
    thresh = threshold_adaptive(gray, 29, offset=5).astype("uint8") * 255
    cv2.imshow("adaptive threshold", thresh)
    # local otsu thresholding
    radius = 5
    selem = disk(radius)
    local_otsu = rank.otsu(blurred, selem)
    grayIm = img_as_ubyte(blurred)
    #cv2.imshow('local otsu', grayIm>=local_otsu)
    plt.imshow(grayIm>=local_otsu, cmap=plt.cm.gray)
    plt.show()
    #thresh2 = threshold_adaptive(local_otsu, 29, offset=5).astype("uint8") * 255
    #cv2.imshow("adaptive threshold after otsu", thresh2)
    cv2.waitKey(0)
    '''
    for body in bods:
        print(body)
        ys = sorted([body[0][1],body[1][1]])
        xs = sorted([body[0][0],body[1][0]])
        crBod = image[ys[0]:ys[1], xs[0]:xs[1]]
        cv2.imshow('cropped', crBod)
        can = imutils.auto_canny(crBod)
        cv2.imshow('canny', can)

        cv2.waitKey(0)
        #exit()
    '''