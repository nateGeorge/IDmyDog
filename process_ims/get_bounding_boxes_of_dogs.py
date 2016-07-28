from __future__ import print_function
import cv2
import os
import numpy as np
import pandas as pd
import random
import pickle as pk
import re
import json

def loadIm(breed, randPic=False):
    # loads image from dataset
    # checks to make sure image is 
    # not none (in case its html, etc)
    global imCnt, pics
    if randPic:
        imPath = random.choice(pics)
        while image==None:
            imPath = random.choice(pics)
            image = cv2.imread(imPath)
    else:
        imPath = pics[imCnt]
        print('pic #', imCnt)
    image = cv2.imread(imPath)
    print('image:', imPath.split('/')[-1])
    clone = image.copy()
    return image, clone, imPath

def nextIm(randPic=False, dir='fwd', newBreed=False):
    # increments through to the next image
    # writes and resets ROIs
    global imCnt, pics
    if randPic:
        image, clone, imPath = loadIm(breed, randPic=True)
    else:
        print(dir)
        if dir=='fwd':
            imCnt += 1
            if imCnt >= len(pics):
                imCnt = 0 # loop back to beginning if at end
            imPath = pics[imCnt]
            image = cv2.imread(imPath)
            while image==None:
                imCnt += 1
                if imCnt >= len(pics):
                    imCnt = 0 # loop back to beginning if at end
                imPath = pics[imCnt]
                image = cv2.imread(imPath)
        elif dir=='bwd':
            imCnt -= 1
            if imCnt < 0:
                imCnt = len(pics) - 1 # go to last pic if at beginning
            imPath = pics[imCnt]
            image = cv2.imread(imPath)
            while image==None:
                imCnt -= 1
                if imCnt < 0:
                    imCnt = len(pics) - 1 # go to last pic if at beginning
                imPath = pics[imCnt]
                image = cv2.imread(imPath)
        if newBreed:
            imCnt = 0
        image, clone, imPath = loadIm(breed)
    print('image data:', pDogs[pDogs.path == imPath])
    return image, clone, imPath

def writeROIs(appendDict, imPath):
    idx = pDogs[pDogs.path == imPath].index[0]
    if appendDict['bodies'] != []:
        pDogs.iloc[idx, 2] = appendDict['bodies']
    if appendDict['heads'] != []:
        pDogs.iloc[idx, 3] = appendDict['heads']
    field = 'bodies'
    appendDict = {}
    appendDict['bodies'] = []
    appendDict['heads'] = []
    return field, appendDict

def sortPics():
    global pics, breed
    pics = pDogs[pDogs['breed'] == breed].path.tolist()
    picNames = [e.split('/')[-1] for e in pics]
    akcPic = max(pics, key=lambda x: len(x)) #akc pic encoded with hash so has longest len
    pics.remove(akcPic)
    picNosRE = [(pics.index(e), re.search('(\d)-(\d)', e).group(1), re.search('(\d)-(\d)', e).group(2)) for e in pics]
    picNosRE.sort(key = lambda x: (x[1], x[2]))
    picsIdxs = [x[0] for x in picNosRE]
    sortedPics = [pics[i] for i in picsIdxs]
    pics = [akcPic] + sortedPics

def getBBs(event, x, y, flags, param):
    # takes BGR CV2 image as an input
    # displays image and waits for
    # bounding boxes to be clicked
    # around dogs' bodies and faces
    # returns np array of bounding boxes
    global pdDict, refPt, field
    
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [(x, y)]
        print('mouse down', x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates and indicate that
        # the cropping operation is finished
        refPt.append((x,y))
        appendDict[field].append(refPt)
        print('mouse up', x, y)
        
        # draw a rectangle around the region of interest
        cv2.rectangle(image, refPt[0], refPt[1], (0, 255, 0), 2)
        cv2.imshow("image", image)

# load configuration
with open('../config.json', 'rb') as f:
    config = json.load(f)

mainImPath = config['image_dir']
pDir = config['pickle_dir']

refPt = []
image = None

breeds = sorted(os.listdir(mainImPath))
breeds.remove('full') # folder generated from Scrapy
breeds.remove('test') # I think I made this folder

# takes a long time to read in images--try to load from pickle if can
try:
    pDogs = pk.load(open(pDir + 'pDogs.pd.pk', 'rb'))
except IOError:
    pDogs = pd.DataFrame(columns=['breed', 'path', 'bodies', 'heads'])
    for breed in breeds:
        breedFolder = mainImPath + breed
        pics = os.listdir(breedFolder)
        for pic in pics:
            if os.path.isfile(breedFolder + '/' + pic):
                pDogs = pDogs.append({'breed':breed, 'path':breedFolder + '/' + pic}, ignore_index=True)
    
    pk.dump(pDogs, open(pDir + 'pDogs.pd.pk', 'wb'))

pDogs = pk.load(open(pDir + 'pDogs-bounding-boxes.pd.pk', 'rb'))
bb = pDogs.dropna()

breedCnt = 0

# this section is for loading data that has only been partially populated
# uncomment if you partially went through and haven't finished
'''
loadedBreeds = sorted(bb.breed.unique())
lastbreed = loadedBreeds[-1]
breedCnt = breeds.index(lastbreed) + 1
'''

cv2.namedWindow("image")
cv2.setMouseCallback("image", getBBs)

imCnt = 0
breed = breeds[breedCnt]
print('breed:', breed)
sortPics()
print(len(pics), '# of pics')
image, clone, imPath = loadIm(breed)
cv2.imshow("image", image)
field = 'bodies'
appendDict = {}
appendDict['bodies'] = []
appendDict['heads'] = []

# keep looping until the 'q' key is pressed
while True:
    # display the image and wait for a keypress
    key = cv2.waitKey(1) & 0xFF
    
    '''
    # for debugging
    if key!=255:
        print(key)
    '''
    
    # if the 'r' key is pressed, reset everything
    if key == ord('r'):
        print('reset')
        print('tracking bodies')
        field = 'bodies'
        appendDict = {}
        appendDict['bodies'] = []
        appendDict['heads'] = []
        image = clone.copy()
        cv2.imshow("image", image)
    
    # if the 'b' key is pressed, log position of dogs' bodies
    if key == ord('b'):
        print('tracking bodies')
        field = 'bodies'
    
    # if the 'f' key is pressed, log position of dogs' faces
    if key == ord('f') or key == ord('a'):
        print('tracking faces')
        field = 'heads'
    
    # if the 'n' key is pressed, go to random dog pic
    if key == ord('n'):
        field, appendDict = writeROIs(appendDict, imPath)
        image, clone, imPath = nextIm(randPic=True)
        cv2.imshow("image", image)
    
    # if fwd (right) arrow pressed go to next in order pic
    if key == 83:
        field, appendDict = writeROIs(appendDict, imPath)
        image, clone, imPath = nextIm(dir='fwd')
        cv2.imshow("image", image)
    
    if key == 81:
        field, appendDict = writeROIs(appendDict, imPath)
        image, clone, imPath = nextIm(dir='bwd')
        cv2.imshow("image", image)
    
    # if the 'd' key is pressed, go to next breed
    # and load next pic
    if key == ord('d'):
        pk.dump(pDogs, open(pDir + 'pDogs-bounding-boxes.pd.pk', 'wb'))
        field, appendDict = writeROIs(appendDict, imPath)
        breedCnt += 1
        if breedCnt >= len(breeds):
            print('reached end of breeds')
            break
        breed = breeds[breedCnt]
        sortPics()
        print('breed: ', breed)
        image, clone, imPath = nextIm(newBreed=True)
        cv2.imshow('image', image)
    
    # if the 'q' key is pressed, break from the loop
    elif key == ord('q'):
        appendDict = writeROIs(appendDict, imPath)
        break

pk.dump(pDogs, open(pDir + 'pDogs-bounding-boxes.pd.pk', 'wb'))