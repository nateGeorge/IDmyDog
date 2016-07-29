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
        breed.append([a.breed] * 13)
        
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

histNtext = pk.load(open(pDir + 'histNtext-fg+bg.pd.pk', 'rb'))
histNtext.reset_index(inplace=True)
hNt = histNtext

# This section was necessary when I forgot to add in the breed information
# the first time working through this.  It shouldn't be necessary now.
if 'breed' not in histNtext.columns:
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
    hNt.drop('index', 1, inplace=True)
    # save to pickle file so we don't have to do this again
    pk.dump(hNt, open(pDir + 'histNtext-fg+bg.pd.pk', 'wb'))

# uncomment to show examples:
show_examples([100, 515, 780])
show_examples([0, 10, 25, 50, 75, 100, 125, 150, 175, 200, 250, 300, 350, 400, 500, 515, 600, 780, 1000, 1200, 1300], printStd=False)

# uncomment to plot foreground haralick standard deviations
get_hara_stats(hNt)

# make dataframes with each component of haralick texture as a column
bgHDF = pd.DataFrame(index=range(hNt.shape[0]), columns=['bg{}'.format(i) for i in range(1,14)] + ['breed'])
fgHDF = pd.DataFrame(index=range(hNt.shape[0]), columns=['fg{}'.format(i) for i in range(1,14)] + ['breed'])

# transform the data by taking the log because it varies over orders of magnitude
# also need to take absolute value because some values are negative
for i in range(hNt.shape[0]):
    for j in range(13):
        bgHDF.iloc[i,j] = np.log(abs(hNt.iloc[i]['bgHaralick'][j]))
        fgHDF.iloc[i,j] = np.log(abs(hNt.iloc[i]['fgHaralick'][j]))
    bgHDF.iloc[i]['breed'] = hNt.iloc[i].breed
    fgHDF.iloc[i]['breed'] = hNt.iloc[i].breed

# fit PCA to the data
pcaBG = PCA(n_components=6)
pcaBG.fit(bgHDF.drop('breed', 1))
pcaFG = PCA(n_components=6)
pcaFG.fit(fgHDF.drop('breed', 1))
varianceBG = pcaBG.explained_variance_ratio_
varianceFG = pcaFG.explained_variance_ratio_
print(varianceBG)
print(varianceFG)
# about 95% of the variance is captured by the first 3 components of PCA
print('top 3 bg components:', varianceBG[:3].sum())
print('top 3 fg components:', varianceFG[:3].sum())
print('top 4 fg components:', varianceFG[:4].sum())

# save pcaFG fit for later use
pk.dump(pcaFG, open(pDir + 'pcaFG.pk', 'wb'))

# plot cumulative distribution of PCA componenents
serBG = pd.Series(varianceBG)
serBG = serBG.cumsum()
serFG = pd.Series(varianceFG)
serFG = serFG.cumsum()
dims = np.array(['Dim{}'.format(i) for i in range(1,7)]*2)
labels = ['foreground']*6 + ['background']*6
bgDF = pd.DataFrame(np.vstack((dims, np.hstack((serFG, serBG)), labels)).T, columns = ['dimension', 'cumulative sum', 'location'])
bgDF['cumulative sum'] = bgDF['cumulative sum'].astype('float64')
'''f, ax = plt.subplots(1,1)
sns.pointplot(x=['Dim{}'.format(i) for i in range(1,7)], y=serBG, color='black', ax=ax, label='background')
g = sns.pointplot(x=, y=serFG, color='red', ax=ax, label='foreground')'''
sns.factorplot(x='dimension', y='cumulative sum', data=bgDF, hue='location')
ax = plt.gca()
ax.set_ylim([0.5,1.05])
ax.set_title('variance contribution of PCA features')
plt.show()


# transform the data with our PCA
reduced_data_BG = pd.DataFrame(np.hstack((pcaBG.transform(bgHDF.drop('breed', 1)), bgHDF.breed[:, np.newaxis])), 
                               columns=['Dim{}'.format(i) for i in range(1,7)] + ['breed'])
reduced_data_FG = pd.DataFrame(np.hstack((pcaBG.transform(fgHDF.drop('breed', 1)), fgHDF.breed[:, np.newaxis])), 
                               columns=['Dim{}'.format(i) for i in range(1,7)] + ['breed'])

# need an 'index' column to be able to merge with breed info
# did this the first time through, shouldn't be necessary now
'''
reduced_data_BG.reset_index(inplace=True)
new_BG = reduced_data_BG.merge(hNt[['index', 'breed']], on='index')
reduced_data_FG.reset_index(inplace=True)
new_FG = reduced_data_FG.merge(hNt[['index', 'breed']], on='index')
'''

# analyze outliers
outliers, outFeats, outDict = getOutliers(reduced_data_FG)
# we only care about outliers in more than one dim
throwOut = []
for k, v in outDict.items():
    if len(v) > 1:
        throwOut.append(k)

reduced_data_FG = reduced_data_FG.drop(reduced_data_FG.index[throwOut]).reset_index(drop = True)

pk.dump(reduced_data_FG, open(pDir + 'training_data-13dimHaraFG-PCA.pd.pk', 'wb'))

# examine a subset of the data
testBreeds = ['Affenpinscher', 'Afghan Hound', 'Norwegian Buhund', 'Czechoslovakian Vlcak', 'Boxer', 'Dogue de Bordeaux']
test_BG = reduced_data_BG[reduced_data_BG.breed=='Lagotto Romagnolo']
for breed in testBreeds:
    test_BG = test_BG.append(reduced_data_BG[reduced_data_BG.breed==breed])

test_FG = reduced_data_FG[reduced_data_FG.breed=='Lagotto Romagnolo']
for breed in testBreeds:
    test_FG = test_FG.append(reduced_data_FG[reduced_data_FG.breed==breed])

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