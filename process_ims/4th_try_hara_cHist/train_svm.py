import cPickle
import pandas as pd
import pickle as pk
import numpy as np
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn.grid_search import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-dark')

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

clf = SVC(random_state=42)
params = {
          'kernel': ['rbf', 'sigmoid'],
          'C': [0.1, 1, 10.0],
          'gamma': ['auto', 1/15., 1/30., 1/45.]
         }

model = GridSearchCV(clf, params, cv=3, refit=False)
model.fit(data, labels)
print("[INFO] best hyperparameters: {}".format(model.best_params_))
model = SVC(C=10, gamma=1/45., random_state=42) # using best params
model.fit(X_train, y_train)

model.score(X_test, y_test)

predictions = model.predict(X_test)

score = 0
for i in range(predictions.shape[0]):
    if predictions[i]==y_test.iloc[i]:
        score+=1

print(score)

# try kNN classifier
from sklearn.neighbors import KNeighborsClassifier
params = {'n_neighbors': range(3,10)}
neigh = KNeighborsClassifier(weights='distance')
model = GridSearchCV(neigh, params, cv=3, refit=False)
model.fit(data, labels)
print("[INFO] best hyperparameters: {}".format(model.best_params_))
model = KNeighborsClassifier(weights='distance', n_neighbors=7)
model.fit(X_train, y_train)
model.score(X_test, y_test)
predictions = model.predict(X_test)

score = 0
for i in range(predictions.shape[0]):
    if predictions[i]==y_test.iloc[i]:
        score+=1

print(score)

# try random forest
from sklearn.ensemble import RandomForestClassifier

params = {
          'max_depth': [20, 30, 40, 50],
          'n_estimators': [30, 40]
         }

clf = RandomForestClassifier(random_state=42, n_jobs=-1)
model = GridSearchCV(clf, params, cv=3, refit=False)
model.fit(data, labels)
print("[INFO] best hyperparameters: {}".format(model.best_params_))

# refit the data from scratch
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

pdRep = pd.DataFrame.from_dict(predDict, orient='index')

# make a 2D dataset to plot heatmap of total images in dataset vs correct predictions
pdRep['total_correct'] = [(pdRep.iloc[i]['total_images'], pdRep.iloc[i]['correct_predictions']) for i in range(pdRep.shape[0])]
TC = pdRep['total_correct'].value_counts()
totalIms = [TC.index[i][0] for i in range(TC.shape[0])]
totalIms = range(max(totalIms) + 1)
correctPreds = [TC.index[i][1] for i in range(TC.shape[0])]
correctPreds = range(max(correctPreds) + 1)
TCheat = np.zeros((len(totalIms), len(correctPreds)))
pdRep.to_csv('report.csv', index_label='breed')
for i, j in TC.index:
    TCheat[i][j] = TC[(i, j)]

sns.heatmap(TCheat) # didn't work too well
plt.show()

sns.lmplot(x='total_images', y='correct_predictions', data=pdRep, fit_reg=False)
#np.vstack((pdRep['correct_predictions'], pdRep['total_images']))

sns.pairplot(pdRep, size=2)
plt.show()

# train model on full training dataset
fullModel = RandomForestClassifier(max_depth=30, n_estimators=40, random_state=42, n_jobs=-1) # best parameters
fullModel.fit(data, labels)
with open('../pickle_files/RTCmodel.pk', 'w') as f:
    f.write(cPickle.dumps(fullModel))

#print(classification_report(y_test, predictions))

# try randomForest with just the 13-dim Haralick features
Hdata = data[:]
for i in range(len(Hdata)):
    Hdata[i] = Hdata[i][-13:] # only keep haralick features

X_train, X_test, y_train, y_test = train_test_split(Hdata, labels, test_size=0.33, stratify=labels, random_state=42)

params = {
          'max_depth': [20, 30, 40, 50],
          'n_estimators': [30, 40]
         }

clf = RandomForestClassifier(random_state=42, n_jobs=-1)
model = GridSearchCV(clf, params, cv=3, refit=False)
model.fit(data, labels)
print("[INFO] best hyperparameters: {}".format(model.best_params_))

model = RandomForestClassifier(max_depth=40, n_estimators=40, random_state=42, n_jobs=-1) # best parameters
model.fit(X_train, y_train)
model.score(X_test, y_test)
predictions = model.predict(X_test)

score = 0

for i in range(predictions.shape[0]):
    if predictions[i]==y_test.iloc[i]:
        score+=1

print(score)
