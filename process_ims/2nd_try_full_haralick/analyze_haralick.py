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

def show_examples(idxs, printStd=True):
    # prints example dataset from supplied indexs, idxs
    # and plots the foreground haralick features
    x = list(range(1,14))
    xs = []
    hara = []
    breed = []
    for idx in idxs:
        a = hNt.iloc[idx]
        xs.append(x)
        hara.append(np.log(abs(a.fgHaralick)))
        breed.append([a.breed]*13)
        
        if printStd:
            print('breed:', a.breed)
            print('filename:', a.file)
            print('foreground Haralick:', a.fgHaralick)
            print('background Haralick:', a.bgHaralick)
    
    newDF = pd.DataFrame(columns=['Haralick feature', 'log(Haralick feature value)', 'breed'])
    newDF['Haralick feature'] = np.array(xs).flatten()
    newDF['log(Haralick feature value)'] = np.array(hara).flatten()
    newDF['breed'] = np.array(breed).flatten()
    newDF.sort_values(by='breed', inplace=True)
    sns.lmplot(x='Haralick feature', y='log(Haralick feature value)', data=newDF, fit_reg=False, hue='breed')
    plt.xticks(x)
    plt.show()

def get_hara_stats(df):
    # gets statistics on haralick features
    # takes dataframe with haralick and breeds
    x = list(range(1,14))
    xs = []
    haraFG = []
    breed = []
    for i in range(df.shape[0]):
        a = df.iloc[i]
        xs.append(x)
        haraFG.append(a.fgHaralick)
        breed.append([a.breed]*13)
    
    newDF = pd.DataFrame(columns=['Haralick feature', 'Haralick feature value', 'breed'])
    newDF['Haralick feature'] = np.array(xs).flatten()
    newDF['Haralick FG feature value'] = np.array(haraFG).flatten()
    newDF['breed'] = np.array(breed).flatten()
    stds = []
    for i in x:
        stds.append(newDF[newDF['Haralick feature']==i]['Haralick FG feature value'].std()
                    / newDF[newDF['Haralick feature']==i]['Haralick FG feature value'].mean())
    
    data = np.vstack((np.array(x), np.array(stds))).T
    pltDF = pd.DataFrame(columns=['Haralick feature', 'relative standard deviation'], data=data)
    sns.lmplot(x='Haralick feature', y='relative standard deviation', data=pltDF, fit_reg=False)
    plt.xticks(x)
    plt.show()

def getOutliers(df):
    # calculates quartiles and gets outliers
    outliers = []
    feats = []
    outlierDict = {}
    # only care about the first 3 dims
    features = ['Dim{}'.format(i) for i in range(1, 4)]
    for feature in features:
        Q1 = np.percentile(df[feature], 25)
        Q3 = np.percentile(df[feature], 75)
        step = (Q3-Q1) * 1.5
        newOutliers = df[~((df[feature] >= Q1 - step) & (df[feature] <= Q3 + step))].index.values
        outliers.extend(newOutliers)
        feats.extend(newOutliers.shape[0] * [feature])
        for out in newOutliers:
            outlierDict.setdefault(out, []).append(feature)
    
    return sorted(list(set(outliers))), zip(outliers, feats), outlierDict

# load configuration
with open('../config.json', 'rb') as f:
    config = json.load(f)

mainImPath = config['image_dir']
pDir = config['pickle_dir']

histNtext = pk.load(open(pDir + 'histNtext.pd.pk', 'rb'))
histNtext.reset_index(inplace=True)

bb = pk.load(open(pDir + 'pDogs-bounding-boxes-clean.pd.pk', 'rb'))
bb.dropna(inplace=True)
bb.reset_index(inplace=True)

# make column of just filename in breed/bounding box DF
files = []
for i in range(bb.shape[0]):
    imName = bb.iloc[i].path.split('/')[-1]
    sb = re.search('St. Bernard', imName)
    ext = re.search('\.\w', imName)
    if ext:
        if sb:
            imName = ' '.join(imName.split('.')[0:2])
        else:
            imName = imName.split('.')[0]
    elif imName[-1]=='.':
        imName = imName[:-1]
    files.append(imName)

bb['raw_file_name'] = pd.Series(files)

# add raw filename column to histogram and haralick texture DF
files = []
for i in range(histNtext.shape[0]):
    files.append(histNtext.iloc[i].file[:-4])

histNtext['raw_file_name'] = pd.Series(files)

# add breed info to histNtext DF
hNt = histNtext.merge(bb[['breed', 'raw_file_name']], on='raw_file_name')
hNt.drop('index', 1, inplace=True)
hNt.reset_index(inplace=True)

# uncomment to show examples:
show_examples([100, 515, 780])
show_examples([0, 10, 25, 50, 75, 100, 125, 150, 175, 200, 250, 300, 350, 400, 500, 515, 600, 780, 1000, 1200, 1300], printStd=False)

# uncomment to plot foreground haralick standard deviations
get_hara_stats(hNt)

# make dataframes with each component of haralick texture as a column
fgHDF = pd.DataFrame(index=range(hNt.shape[0]), columns=['breed'] + ['fg{}'.format(i) for i in range(1,14)])

for i in range(hNt.shape[0]):
    for j in range(13):
        fgHDF.iloc[i,j + 1] = np.log(abs(hNt.iloc[i]['fgHaralick'][j]))
    fgHDF.iloc[i, 0] = hNt.iloc[i].breed

pk.dump(fgHDF, open(pDir + 'fgHDF-full.pd.pk', 'wb'))
exit()

'''
# analyze outliers
outliers, outFeats, outDict = getOutliers(reduced_data_FG)
# we only care about outliers in more than one dim
throwOut = []
for k, v in outDict.items():
    if len(v) > 1:
        throwOut.append(k)

new_FG = new_FG.drop(reduced_data_FG.index[throwOut]).reset_index(drop = True)

pk.dump(new_FG, open('pickle_files/training_data.pd.pk', 'wb'))
'''

# examine a subset of the data
testBreeds = ['Affenpinscher', 'Afghan Hound', 'Norwegian Buhund', 'Czechoslovakian Vlcak', 'Boxer', 'Dogue de Bordeaux']
test_BG = new_BG[new_BG.breed=='Lagotto Romagnolo']
for breed in testBreeds:
    test_BG = test_BG.append(new_BG[new_BG.breed==breed])

test_FG = new_FG[new_FG.breed=='Lagotto Romagnolo']
for breed in testBreeds:
    test_FG = test_FG.append(new_FG[new_FG.breed==breed])

sns.pairplot(test_BG[['Dim{}'.format(i) for i in range(1,4)] + ['breed']], hue='breed')
plt.show()

sns.pairplot(test_FG[['Dim{}'.format(i) for i in range(1,4)] + ['breed']], hue='breed')
plt.show()

# generate pairplot of foreground haralick PCA components 1-3
test_FG = test_FG.sort_values(by='breed')
g = sns.PairGrid(test_FG[['Dim{}'.format(i) for i in range(1,4)] + ['breed']], hue='breed')
g = g.map_diag(sns.kdeplot)
g = g.map_upper(plt.scatter)
g = g.map_lower(sns.kdeplot, cmap='Blues_d')
g.add_legend()
plt.show()