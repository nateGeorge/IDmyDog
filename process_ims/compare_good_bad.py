import seaborn as sns
import numpy as np
import cv2
import os
import pandas as pd
import pickle
import pylab as plt
import mahotas as mh
from sklearn.svm import LinearSVC, SVC

# load hand-picked good images in 'train' folders 
try:
    breed_ims_good = pickle.load(open('pickle_files/breed_ims-aff-afg-good.pd.pk', 'rb'))
except IOError:
    imPath = '/media/nate/Windows/github/IDmyDog/scrape-ims/images/'
    breeds = os.listdir(imPath)[:2] # only use the first two breeds for now
    print(breeds)   
    
    pdDict = {}
    pdDict['picPaths'] = []
    pdDict['images'] = []
    pdDict['breed_labels'] = []
    for breed in breeds:
        breedFolder = imPath + breed
        pics = os.listdir(breedFolder + '/train')
        for pic in pics:
            picPath = breedFolder + '/' + pic
            print(breed, pic)
            image = cv2.imread(picPath)
            if image!=None: # some image links redirected to a 404, etc
                pdDict['picPaths'].append(picPath)
                pdDict['images'].append(cv2.imread(picPath))
                pdDict['breed_labels'].append(breed)
    
    breed_ims_good = pd.DataFrame(pdDict)
    
    pickle.dump(breed_ims_good, open('pickle_files/breed_ims-aff-afg-good.pd.pk', 'wb'))

# load hand-picked bad images in 'complex' folders
try:
    breed_ims_bad = pickle.load(open('pickle_files/breed_ims-aff-afg-bad.pd.pk', 'rb'))
except IOError:
    imPath = '/media/nate/Windows/github/IDmyDog/scrape-ims/images/'
    breeds = os.listdir(imPath)[:2] # only use the first two breeds for now
    print(breeds)   
    
    pdDict = {}
    pdDict['picPaths'] = []
    pdDict['images'] = []
    pdDict['breed_labels'] = []
    for breed in breeds:
        breedFolder = imPath + breed
        pics = os.listdir(breedFolder + '/complex')
        for pic in pics:
            picPath = breedFolder + '/' + pic
            print(breed, pic)
            image = cv2.imread(picPath)
            if image!=None: # some image links redirected to a 404, etc
                pdDict['picPaths'].append(picPath)
                pdDict['images'].append(cv2.imread(picPath))
                pdDict['breed_labels'].append(breed)
    
    breed_ims_bad = pd.DataFrame(pdDict)
    
    pickle.dump(breed_ims_bad, open('pickle_files/breed_ims-aff-afg-bad.pd.pk', 'wb'))
