import cPickle
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import pickle as pk
import numpy as np
import glob
import os
import cv2
import imutils
import json
from mahotas.features import haralick
from sklearn.decomposition import PCA

def get_color_hists(im):
    # returns normalized histograms of color for image im
    b, g, r = cv2.split(im)
    fgHist = {}
    for chan, col in zip([r, g, b], ['red', 'green', 'blue']):
        fgHist[col] = cv2.calcHist([chan], [0], None, [256], [0, 256])
        fgHist[col] /= fgHist[col].sum() # normalize to compare images of different sizes
    
    fullHist = np.hstack((fgHist['blue'][:,0], fgHist['green'][:,0], fgHist['red'][:,0]))
    fullHist = pcaFG.transform(fullHist.reshape(1, -1))
    return fullHist

def get_features(im):
    # gets haralick and color histogram PCA of image
    h, w = im.shape[:2]
    if w > 450:
        im = imutils.resize(im, width=450)
        h, w = im.shape[:2]
    im = im[h/5:-h/5, w/5:-w/5] # only use center 60% of image to avoid background noise
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    hara = haralick(gray).flatten()
    cHist = get_color_hists(im)
    return np.hstack((hara, cHist.T[:,0]))

# load configuration
with open('../config.json', 'rb') as f:
    config = json.load(f)

mainImPath = config['image_dir']
pDir = config['pickle_dir']

bb = pk.load(open(pDir + 'pDogs-bounding-boxes-clean.pd.pk', 'rb'))
bb.dropna(inplace=True)

breeds = sorted(bb.breed.unique().tolist())

pcaFG = pk.load(open(pDir + 'pcaFG20.pd.pk', 'rb'))

dataDict = {}
testIms = 3
np.random.seed(42)
# grab three images from each breed, making sure they aren't in the training set
print('[INFO] loading images')
for breed in breeds:
    train_ims = bb[bb.breed==breed].path
    folder = mainImPath + breed + '/'
    cnt = 0
    things = os.listdir(folder)
    np.random.shuffle(things)
    for thing in things:
        thing = folder + thing
        if os.path.isfile(thing) and thing not in train_ims and cnt < testIms:
            try:
                im = cv2.imread(thing)
            except: # sometimes throws UnicodeEncodeError or libpng warning for some reason
                continue
            if im==None: # some images are actually html file from a redirect
                continue
            feats = get_features(im)
            if isinstance(feats, float): # sometimes returns nan, don't know why
                continue
            key = thing.split('/')[-1] # only save the filename as dict key
            dataDict[key] = {}
            dataDict[key]['file'] = thing
            dataDict[key]['breed'] = breed
            dataDict[key]['features'] = feats
            cnt += 1

print('[INFO]: total images grabbed:', len(dataDict))
dataDF = pd.DataFrame.from_dict(dataDict, orient='index')

# load RandomForest model
print('[INFO] loading ML model')
with open(pDir + 'RTCmodel.pk', 'r') as f:
    fullModel = cPickle.loads(f.read())

data = dataDF.features.tolist()
labels = dataDF.breed.tolist()

predictions = fullModel.predict(data)

score = 0

for i in range(predictions.shape[0]):
    if predictions[i]==labels[i]:
        score+=1

print('[INFO] model score:')
print(score)
print(fullModel.score(data, labels))