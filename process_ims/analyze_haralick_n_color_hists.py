from __future__ import print_function
from sklearn.decomposition import PCA
import pandas as pd
import pickle as pk
import re
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-dark')

def show_examples(idxs, printStd=True):
    x = ['#{}'.format(i) for i in range(1,14)]
    x = list(range(1,14))
    xs = []
    hara = []
    breed = []
    noDF = True
    for idx in idxs:
        a = hNt.iloc[idx]
        xs.append(x)
        hara.append(np.log(abs(a.fgHaralick)))
        breed.append([a.breed]*13)
        if noDF:
            df = a
            noDF = False
        else:
            df = df.append(a)
        
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


histNtext = pk.load(open('pickle_files/histNtext.pd.pk', 'rb'))
histNtext.reset_index(inplace=True)

bb = pk.load(open('pickle_files/pDogs-bounding-boxes-clean.pd.pk', 'rb'))
bb.dropna(inplace=True)
bb.reset_index(inplace=True)


# make column of just filename in breed/bounding box DF
mainImPath = '/media/nate/Windows/github/IDmyDog/scrape-ims/images/'
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
    if imName == 'd8a75cb8ad0baa572575c9909fcb6901f30dea8b':
        print(imName, i)
    files.append(imName)

bb['raw_file_name'] = pd.Series(files)

# add raw filename column to histogram and haralick texture DF
files = []
for i in range(histNtext.shape[0]):
    files.append(histNtext.iloc[i].file[:-4])

histNtext['raw_file_name'] = pd.Series(files)

# add breed info to histNtext DF
hNt = histNtext.merge(bb[['breed', 'raw_file_name']], on='raw_file_name')

# uncomment to show examples:
#show_examples([100, 515, 780])
#show_examples([0, 10, 25, 50, 75, 100, 125, 150, 175, 200, 250, 300, 350, 400, 500, 515, 600, 780, 1000, 1200, 1300], printStd=False)

# make dataframes with each component of haralick texture as a column
bgHDF = pd.DataFrame(index=range(hNt.shape[0]), columns=['bg{}'.format(i) for i in range(1,14)])
fgHDF = pd.DataFrame(index=range(hNt.shape[0]), columns=['fg{}'.format(i) for i in range(1,14)])

# transform the data by taking the log because it varies over orders of magnitude
# also need to take absolute value because some values are negative
for i in range(histNtext.shape[0]):
    for j in range(fullDF.shape[1]):
        bgHDF.iloc[i,j] = np.log(abs(hNt.iloc[i]['bgHaralick'][j]))
        fgHDF.iloc[i,j] = np.log(abs(hNt.iloc[i]['fgHaralick'][j]))

# fit PCA to the data
pcaBG = PCA(n_components=6)
pcaBG.fit(bgHDF)
pcaFG = PCA(n_components=6)
pcaFG.fit(fgHDF)
varianceBG = pcaBG.explained_variance_ratio_
varianceFG = pcaFG.explained_variance_ratio_
print(varianceBG)
print(varianceFG)
# about 95% of the variance is captured by the first 3 components of PCA
print('top 3 bg components:', varianceBG[:3].sum())
print('top 3 bg components:', varianceFG[:3].sum())

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
reduced_data_BG = pd.DataFrame(pcaBG.transform(bgHDF), columns=['Dim{}'.format(i) for i in range(1,7)])
reduced_data_FG = pd.DataFrame(pcaBG.transform(bgHDF), columns=['Dim{}'.format(i) for i in range(1,7)])

# need an 'index' column to be able to merge with breed info
reduced_data_BG.reset_index(inplace=True)
new_BG = reduced_data_BG.merge(hNt[['index', 'breed']], on='index')
reduced_data_FG.reset_index(inplace=True)
new_FG = reduced_data_FG.merge(hNt[['index', 'breed']], on='index')

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
g = sns.PairGrid(test_FG[['Dim{}'.format(i) for i in range(1,4)] + ['breed']], hue='breed')
g.map_diag(sns.kdeplot)
g = g.map_upper(plt.scatter)
g = g.map_lower(sns.kdeplot, cmap='Blues_d')
g.add_legend()
plt.show()



