from __future__ import print_function
import pandas as pd
import pickle as pk
import numpy as np
import json
import cPickle
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn.grid_search import GridSearchCV
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('seaborn-dark')

def get_score(model):
    predictions = model.predict(X_test)
    
    score = 0
    for i in range(predictions.shape[0]):
        if predictions[i]==y_test.iloc[i]:
            score+=1
    print(model.score(X_test, y_test))
    print(score)

# load configuration
with open('../config.json', 'rb') as f:
    config = json.load(f)

mainImPath = config['image_dir']
pDir = config['pickle_dir']

# load pca fit
pcaFG = pk.load(open(pDir + 'pcaFG.pk', 'rb'))
# load training data
print('[INFO] using 3-dim Haralick PCA training data')
training_data = pk.load(open(pDir + 'training_data-13dimHaraFG-PCA.pd.pk', 'rb'))

Tdata = training_data.drop('breed', 1)
data = []
for i in range(Tdata.shape[0]):
    # only use the first 3 components
    data.append([i for i in Tdata.iloc[i, :3]])

labels = training_data.breed

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.33, stratify=labels, random_state=42)


# first try of SVC
clf = SVC(random_state=42)
params = {
          'C': [0.05, 0.1, 0.2, 0.5, 1.0, 2.0],
         }
print('[INFO] SVC gridsearching', params)
model = GridSearchCV(clf, params, cv=3, refit=False)
model.fit(data, labels)
print('[INFO] best hyperparameters: {}'.format(model.best_params_))
model = SVC(C=0.05, random_state=42) # best params
model.fit(X_train, y_train)
print('[INFO] svc score:')
get_score(model)

# try further refinement of the SVC
params = {
          'kernel': ['rbf', 'sigmoid'],
          'C': [0.05, 0.1, 0.2, 0.5, 1.0, 2.0],
          'gamma': ['auto', 1, 1/2.]
         }
print('[INFO] SVC gridsearching', params)
model = GridSearchCV(clf, params, cv=3, refit=False)
model.fit(data, labels)
print('[INFO] best hyperparameters: {}'.format(model.best_params_))
model = SVC(C=0.5, gamma=1, random_state=42) # best params
model.fit(X_train, y_train)
print('[INFO] svc score:')
get_score(model)

# try kNN classifier
params = {
            'weights': ['distance', 'uniform'],
            'n_neighbors': range(3,10)
         }
print('[INFO] kNN gridsearching', params)
neigh = KNeighborsClassifier()
model = GridSearchCV(neigh, params, cv=3, refit=False)
model.fit(data, labels)
print('[INFO] best hyperparameters: {}'.format(model.best_params_))
model = KNeighborsClassifier(weights='uniform', n_neighbors=3) # best params found
model.fit(X_train, y_train)
print('[INFO] kNN score:')
get_score(model)

# try random forest
params = {
          'max_depth': [20, 30, 40, 50, 60],
          'n_estimators': [30, 40]
         }
print('[INFO] RandomForest gridsearching', params)
clf = RandomForestClassifier(random_state=42, n_jobs=-1)
model = GridSearchCV(clf, params, cv=3, refit=False)
model.fit(data, labels)
print('[INFO] best hyperparameters: {}'.format(model.best_params_))
model = RandomForestClassifier(max_depth=50, n_estimators=30, random_state=42, n_jobs=-1) # best parameters
model.fit(X_train, y_train)
print('[INFO] RandomForest score:')
get_score(model)

# try using the 13-dim haralick features instead of PCA
# load training data
print('[INFO] using 13-dim Haralick features training data')
training_data = pk.load(open(pDir + 'fgHDF-full.pd.pk', 'rb'))

Tdata = training_data.drop('breed', 1)
data = []
for i in range(Tdata.shape[0]):
    data.append([i for i in Tdata.iloc[i]])

labels = training_data.breed

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.33, stratify=labels, random_state=42)

# try SVC again
clf = SVC(random_state=42)
params = {
          'C': [0.05, 0.1, 0.2, 0.5, 1.0, 2.0],
          'gamma': ['auto', 1, 1/5., 1/10.]
         }
print('[INFO] SVC gridsearching', params)
model = GridSearchCV(clf, params, cv=3, refit=False)
model.fit(data, labels)
print('[INFO] best hyperparameters: {}'.format(model.best_params_))
model = SVC(C=1, gamma=1, random_state=42) # best params
model.fit(X_train, y_train)
print('[INFO] svc score:')
get_score(model)

# allow more room for C
params = {
          'C': [0.1, 0.5, 1.0],
          'gamma': ['auto', 1, 1/5., 1/10.]
         }
print('[INFO] SVC gridsearching', params)
model = GridSearchCV(clf, params, cv=3, refit=False)
model.fit(data, labels)
print('[INFO] best hyperparameters: {}'.format(model.best_params_))
model = SVC(C=5, gamma=1, random_state=42) # best params
model.fit(X_train, y_train)
print('[INFO] svc score:')
get_score(model)

# try kNN again
params = {
            'weights': ['distance', 'uniform'],
            'n_neighbors': range(3,10)
         }
print('[INFO] kNN gridsearching', params)
neigh = KNeighborsClassifier()
model = GridSearchCV(neigh, params, cv=3, refit=False)
model.fit(data, labels)
print('[INFO] best hyperparameters: {}'.format(model.best_params_))
model = KNeighborsClassifier(weights='distance', n_neighbors=6) # best params found
model.fit(X_train, y_train)
print('[INFO] kNN score:')
get_score(model)

# try random forest again
params = {
          'max_depth': [20, 30, 40, 50, 60],
          'n_estimators': [30, 40]
         }
print('[INFO] RandomForest gridsearching', params)
clf = RandomForestClassifier(random_state=42, n_jobs=-1)
model = GridSearchCV(clf, params, cv=3, refit=False)
model.fit(data, labels)
print('[INFO] best hyperparameters: {}'.format(model.best_params_))
model = RandomForestClassifier(max_depth=40, n_estimators=40, random_state=42, n_jobs=-1) # best parameters
model.fit(X_train, y_train)
print('[INFO] RandomForest score:')
get_score(model)

# now try the full 13x4 Haralick features
# load training data
print('[INFO] using 52-dim Haralick features')
training_data = pk.load(open(pDir + 'fgHara-full-13x4.pd.pk', 'rb'))

data = []
for i in range(training_data.shape[0]):
    data.append(training_data.iloc[i].fgHaralick.flatten())

labels = training_data.breed

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.33, stratify=labels, random_state=42)

# try SVC again
clf = SVC(random_state=42)
params = {
          'C': [0.05, 0.1, 0.2, 0.5, 1.0, 2.0],
          'gamma': ['auto', 1, 1/10., 1/25.]
         }
print('[INFO] SVC gridsearching', params)
model = GridSearchCV(clf, params, cv=3, refit=False)
model.fit(data, labels)
print('[INFO] best hyperparameters: {}'.format(model.best_params_))
model = SVC(C=1, random_state=42) # best params
model.fit(X_train, y_train)
print('[INFO] svc score:')
get_score(model)

# try kNN again
params = {
            'weights': ['distance', 'uniform'],
            'n_neighbors': range(3,10)
         }
print('[INFO] kNN gridsearching', params)
neigh = KNeighborsClassifier()
model = GridSearchCV(neigh, params, cv=3, refit=False)
model.fit(data, labels)
print('[INFO] best hyperparameters: {}'.format(model.best_params_))
model = KNeighborsClassifier(weights='uniform', n_neighbors=8) # best params found
model.fit(X_train, y_train)
print('[INFO] kNN score:')
get_score(model)

# try random forest again
params = {
          'max_depth': [20, 30, 40, 50, 60],
          'n_estimators': [30, 40]
         }
print('[INFO] RandomForest gridsearching', params)
clf = RandomForestClassifier(random_state=42, n_jobs=-1)
model = GridSearchCV(clf, params, cv=3, refit=False)
model.fit(data, labels)
print('[INFO] best hyperparameters: {}'.format(model.best_params_))
model = RandomForestClassifier(max_depth=50, n_estimators=30, random_state=42, n_jobs=-1) # best parameters
model.fit(X_train, y_train)
print('[INFO] RandomForest score:')
get_score(model)

# now try the 52-dim haralick with 20-dim color histogram PCA
# load training data
print('[INFO] using 52-dim Haralick features and 20-dim color histogram PCA')
training_data1 = pk.load(open(pDir + 'fgHara-full-13x4.pd.pk', 'rb'))
training_data2 = pk.load(open(pDir + 'pcaHists.pd.pk', 'rb'))
training_data2.drop('breed', 1, inplace=True)
training_data = training_data1.merge(training_data2, on='file')

data = []
for i in range(training_data.shape[0]):
    data.append(np.hstack(([training_data.iloc[i]['dim{}'.format(j)] for j in range(20)], training_data.iloc[i].fgHaralick.flatten())))

labels = training_data.breed

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.33, stratify=labels, random_state=42)

# try SVC again
clf = SVC(random_state=42)
params = {
          'C': [1.0, 2.0, 5.0, 10.0],
          'gamma': ['auto', 1, 1/25., 1/50.]
         }
print('[INFO] SVC gridsearching', params)
model = GridSearchCV(clf, params, cv=3, refit=False)
model.fit(data, labels)
print('[INFO] best hyperparameters: {}'.format(model.best_params_))
model = SVC(C=2, random_state=42) # best params
model.fit(X_train, y_train)
print('[INFO] svc score:')
get_score(model)

# try kNN again
params = {
            'weights': ['distance', 'uniform'],
            'n_neighbors': range(3,10)
         }
print('[INFO] kNN gridsearching', params)
neigh = KNeighborsClassifier()
model = GridSearchCV(neigh, params, cv=3, refit=False)
model.fit(data, labels)
print('[INFO] best hyperparameters: {}'.format(model.best_params_))
model = KNeighborsClassifier(weights='uniform', n_neighbors=8) # best params found
model.fit(X_train, y_train)
print('[INFO] kNN score:')
get_score(model)

# try random forest again
params = {
          'max_depth': [20, 30, 40, 50, 60],
          'n_estimators': [30, 40]
         }
print('[INFO] RandomForest gridsearching', params)
clf = RandomForestClassifier(random_state=42, n_jobs=-1)
model = GridSearchCV(clf, params, cv=3, refit=False)
model.fit(data, labels)
print('[INFO] best hyperparameters: {}'.format(model.best_params_))
model = RandomForestClassifier(max_depth=30, n_estimators=40, random_state=42, n_jobs=-1) # best parameters
model.fit(X_train, y_train)
print('[INFO] RandomForest score:')
get_score(model)

model = RandomForestClassifier(max_depth=50, n_estimators=40, random_state=42, n_jobs=-1) # empirically found best parameters
model.fit(X_train, y_train)
print('[INFO] RandomForest score with max_depth = 50:')
get_score(model)

predictions = model.predict(X_test)

predDict = {}
for breed in labels.unique():
    predDict[breed] = {}
    predDict[breed]['correct_predictions'] = 0
    predDict[breed]['total_in_test'] = 0
    predDict[breed]['test_predictions'] = 0
    predDict[breed]['total_images'] = labels[labels==breed].count()

for i in range(predictions.shape[0]):
    predDict[y_test.iloc[i]]['total_in_test'] += 1
    predDict[predictions[i]]['test_predictions'] += 1
    if predictions[i]==y_test.iloc[i]:
        predDict[predictions[i]]['correct_predictions'] += 1

pdRep = pd.DataFrame.from_dict(predDict, orient='index')

sns.lmplot(x='total_images', y='correct_predictions', data=pdRep, fit_reg=False)
#np.vstack((pdRep['correct_predictions'], pdRep['total_images']))

sns.pairplot(pdRep, size=2)
plt.show()

print('[INFO] fitting full model and saving to disk')
fullModel = RandomForestClassifier(max_depth=50, n_estimators=40, random_state=42, n_jobs=-1) # best parameters
fullModel.fit(data, labels)
with open(pDir + 'RTCmodel.pk', 'w') as f:
    f.write(cPickle.dumps(fullModel))
