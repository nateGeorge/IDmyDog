from sklearn.ensemble import RandomForestClassifier
import cPickle
import pandas as pd
import pickle as pk
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-dark')

def add_noise(data, pctNoise=0.10):
    # adds gaussian noise to data
    # designed for 1D vector
    newData = list(data) # make a copy of list
    for i in range(len(newData)):
        noise = np.random.normal()*pctNoise*newData[i]
        newData[i] += noise
    
    return newData

# load training data
hara1 = pk.load(open('../pickle_files/fgHara-full-13x4.pd.pk', 'rb'))
hara2 = pk.load(open('../pickle_files/fgHara-full-13x4-new-new-new.pd.pk', 'rb'))
hara = hara1.append(hara2)
hara.reset_index(inplace=True, drop=True)
hara['meanHaralick'] = hara.fgHaralick.apply(lambda x: x.mean(0))
pcaHists = pk.load(open('../pickle_files/pcaHists.pd.pk', 'rb'))

fullData = pcaHists.merge(hara[['meanHaralick', 'file']], on='file')

data = []
for i in range(fullData.shape[0]):
    # only use the first 3 components
    data.append(np.hstack(([fullData.iloc[i]['dim{}'.format(j)] for j in range(20)], fullData.iloc[i].meanHaralick)))

labels = fullData.breed

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.33, stratify=labels, random_state=42)

model = RandomForestClassifier(max_depth=40, n_estimators=40, random_state=42, n_jobs=-1) # best parameters
model.fit(X_train, y_train)
model.score(X_test, y_test)
predictions = model.predict(X_test)

for k,v in model2.get_params().items():
    print(k, v)
    print(model.estimator.get_params()[k])

predDict = {}
for breed in labels.unique():
    predDict[breed] = {}
    predDict[breed]['correct_predictions'] = 0
    predDict[breed]['total_in_test'] = 0
    predDict[breed]['test_predictions'] = 0
    predDict[breed]['total_images'] = labels[labels==breed].count()

score = 0

for i in range(predictions.shape[0]):
    predDict[y_test.iloc[i]]['total_in_test'] += 1
    predDict[predictions[i]]['test_predictions'] += 1
    if predictions[i]==y_test.iloc[i]:
        score+=1
        predDict[predictions[i]]['correct_predictions'] += 1

print(score)

# check what adding 10% noise to the training data does
from sklearn.cross_validation import train_test_split

hara1 = pk.load(open('../pickle_files/fgHara-full-13x4.pd.pk', 'rb'))
hara2 = pk.load(open('../pickle_files/fgHara-full-13x4-new-new-new.pd.pk', 'rb'))
hara = hara1.append(hara2)
hara.reset_index(inplace=True, drop=True)
hara['meanHaralick'] = hara.fgHaralick.apply(lambda x: x.mean(0))
pcaHists = pk.load(open('../pickle_files/pcaHists.pd.pk', 'rb'))

fullData = pcaHists.merge(hara[['meanHaralick', 'file']], on='file')

data = []
for i in range(fullData.shape[0]):
    # only use the first 3 components
    data.append(np.hstack(([fullData.iloc[i]['dim{}'.format(j)] for j in range(20)], fullData.iloc[i].meanHaralick)))

labels = fullData.breed

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.33, stratify=labels, random_state=42)

np.random.seed(42)
for pct in [0.1, 0.2, 0.3, 0.4]:
    X_train_noisy = X_train[:] # copy list
    for i in range(len(X_train_noisy)):
        X_train_noisy[i] = add_noise(X_train_noisy[i], pctNoise=pct)
    
    model = RandomForestClassifier(max_depth=40, n_estimators=40, random_state=42, n_jobs=-1) # best parameters
    model.fit(X_train_noisy, y_train)
    model.score(X_test, y_test)
    predictions = model.predict(X_test)
    
    score = 0
    
    for i in range(predictions.shape[0]):
        if predictions[i]==y_test.iloc[i]:
            score+=1
    
    print('score with {}% noise:'.format(pct), score)


for pct in [0.1, 0.2, 0.3, 0.4]:
    X_test_noisy = X_test[:] # copy list
    for i in range(len(X_test_noisy)):
        X_test_noisy[i] = add_noise(X_test_noisy[i], pctNoise=pct)
    
    model = RandomForestClassifier(max_depth=40, n_estimators=40, random_state=42, n_jobs=-1) # best parameters
    model.fit(X_train, y_train)
    model.score(X_test_noisy, y_test)
    predictions = model.predict(X_test_noisy)
    
    score = 0
    
    for i in range(predictions.shape[0]):
        if predictions[i]==y_test.iloc[i]:
            score+=1
    
    print('score with {}% noise:'.format(pct), score)
