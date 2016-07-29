# does PCA on color histograms
from __future__ import print_function
from sklearn.decomposition import PCA
import pandas as pd
import pickle as pk
import re
import json
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-dark')

# load configuration
with open('../config.json', 'rb') as f:
    config = json.load(f)

mainImPath = config['image_dir']
pDir = config['pickle_dir']

hists = pk.load(open(pDir + 'fgHists.pd.pk', 'rb'))
fullHists = []
for i in range(hists.shape[0]):
    # flatten histogram
    fullHists.append(np.hstack((hists.iloc[i].fgHist['blue'][:,0], hists.iloc[i].fgHist['green'][:,0], hists.iloc[i].fgHist['red'][:,0])))

iHists = pd.DataFrame(columns = ['bin{}'.format(i) for i in range(len(fullHists[0]))], data=fullHists)
iHists['breed'] = hists.breed
iHists['file'] = hists.file

removeIdx = []
# some values are NaN for some reason, need to remove them
for i in range(iHists.shape[0]):
    if sum(iHists.iloc[i].isnull()) > 0:
        removeIdx.append(i)

newIhists = iHists.drop(removeIdx)
tempH = newIhists.drop('breed', 1)
tempH = tempH.drop('file', 1)

pcaFG = PCA(n_components=20)
pcaFG.fit(tempH)
varianceFG = pcaFG.explained_variance_ratio_
print(varianceFG)
print('top 20 fg components:', varianceFG[:20].sum())

pcaHists = pd.DataFrame(pcaFG.transform(tempH), columns=['dim{}'.format(i) for i in range(20)])
pcaHists['breed'] = newIhists.breed
pcaHists['file'] = newIhists.file

pk.dump(pcaFG, open(pDir + 'pcaFG20.pd.pk', 'wb'))

pk.dump(pcaHists, open(pDir + 'pcaHists.pd.pk', 'wb'))