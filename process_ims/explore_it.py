import seaborn as sns
import numpy as np
import cv2
import os
import pandas as pd
import pickle
import pylab as plt

# load images from pickle otherwise 
try:
    breed_ims = pickle.load(open('breed_ims.pd.pk', 'rb'))
except IOError:
    imPath = '/media/nate/Windows/github/IDmyDog/scrape-ims/images/'
    breeds = os.listdir(imPath)
    print(breeds)
    
    pdDict = {}
    pdDict['picPaths'] = []
    pdDict['images'] = []
    pdDict['breed_labels'] = []
    for breed in breeds:
        breedFolder = imPath + breed
        pics = os.listdir(breedFolder)
        for pic in pics:
            picPath = breedFolder + '/' + pic
            print(breed, pic)
            pdDict['picPaths'].append(picPath)
            pdDict['images'].append(cv2.imread(picPath))
            pdDict['breed_labels'].append(breed)
    
    breed_ims = pd.DataFrame(pdDict)
    
    pickle.dump(breed_ims, open('breed_ims.pd.pk', 'wb'))

# if not done downloading images, only want breeds with many ims
breed_data = []
breed_target = []

imsCnt = breed_ims['breed_labels'].value_counts()
highIms = [imsCnt.index[i] for i in range(len(imsCnt)) if imsCnt[i] > 10]
for i in range(breed_ims.images.shape[0]):
    # todo; try except for nonetype object
    if breed_ims['breed_labels'][i] in highIms:
        breed_data.append(breed_ims.images[i].flatten())
        breed_target.append(breed_ims['breed_labels'][i])

# introspect the images arrays to find the shapes (for plotting)
n_samples, h, w = breed_ims.images.shape
np.random.seed(42)



minH = np.inf
minW = np.inf
Hs = []
Ws = []

for i in range(breed_ims.images.shape[0]):
    try:
        h, w, _ = breed_ims.images[i].shape
        Hs.append(h)
        Ws.append(w)
        if h < minH:
            minH = h
        if w < minW:
            minW = w
    except AttributeError as e:
        print(e)
        print(breed_ims.picPaths[i])
        
sns.distplot(Ws)
plt.show()
sns.distplot(Hs)
plt.show()