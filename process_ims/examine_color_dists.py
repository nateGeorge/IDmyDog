from __future__ import print_function
from sklearn.decomposition import PCA
import pandas as pd
import pickle as pk
import re
import seaborn as sns
import numpy as np
from scipy.stats import kendalltau
import matplotlib.pyplot as plt
plt.style.use('seaborn-dark')

def getOutliers(df):
    # calculates quartiles and gets outliers
    outliers = []
    feats = []
    outlierDict = {}
    # only care about the first 3 dims
    features = ['dim{}'.format(i) for i in range(1, 20)]
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


pcaHists = pk.load(open('../pickle_files/pcaHists.pd.pk', 'rb'))

breedGrp = pcaHists.groupby('breed')

breedDF = breedGrp.mean()

sns.jointplot(x='dim0', y='dim1', data=breedDF, kind="scatter", color="#4CB391")
plt.show()

sns.jointplot(x='dim0', y='dim1', data=breedDF, kind="hex", color="#4CB391")
plt.show()

for i in range(19):
    g = sns.FacetGrid(breedDF, hue='breed')
    g.map(plt.scatter, 'dim{}'.format(i), 'dim{}'.format(i + 1), alpha=.7)
    plt.show()


breedDF.reset_index(inplace=True)
g = sns.FacetGrid(breedDF.iloc[:20], hue='breed')
g.map(plt.scatter, 'dim0', 'dim1', alpha=.7)
plt.show()

g = sns.FacetGrid(pcaHists[:100], hue='breed')
g.map(plt.scatter, 'dim0', 'dim1', alpha=.7)
g.add_legend()
plt.show()

g = sns.FacetGrid(pcaHists[:12], hue='breed')
g.map(plt.scatter, 'dim0', 'dim1', alpha=.7)
g.add_legend()
plt.show()

#dim2 has some large outliers
sns.jointplot(x='dim1', y='dim2', data=breedDF, kind="scatter", color="#4CB391")
plt.show()

outliers, outFeats, outDict = getOutliers(breedDF)

d12out = [(k, v) for k, v in outFeats if v=='dim1']

for breed, dim in d12out:
    print(breed, breedDF[breedDF.breed==breed]['dim1'])

sorted(outDict.items(), key=lambda x: len(x[1]), reverse=True)
