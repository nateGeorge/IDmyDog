from sklearn.ensemble import RandomForestClassifier
import cPickle
import json
import pandas as pd
import pickle as pk
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-dark')

def get_score(model, X_test):
    predictions = model.predict(X_test)
    
    score = 0
    for i in range(predictions.shape[0]):
        if predictions[i]==y_test.iloc[i]:
            score+=1
    print(model.score(X_test, y_test))
    print(score)

def add_noise(data, pctNoise=0.10):
    # adds gaussian noise to data
    # designed for 1D vector
    newData = list(data) # make a copy of list
    for i in range(len(newData)):
        noise = np.random.normal()*pctNoise*newData[i]
        newData[i] += noise
    
    return newData

# load configuration
with open('../config.json', 'rb') as f:
    config = json.load(f)

mainImPath = config['image_dir']
pDir = config['pickle_dir']

# load training data
print('[INFO] loading training data')
training_data1 = pk.load(open(pDir + 'fgHara-full-13x4.pd.pk', 'rb'))
training_data2 = pk.load(open(pDir + 'pcaHists.pd.pk', 'rb'))
training_data2.drop('breed', 1, inplace=True)
training_data = training_data1.merge(training_data2, on='file')

data = []
for i in range(training_data.shape[0]):
    data.append(np.hstack(([training_data.iloc[i]['dim{}'.format(j)] for j in range(20)], training_data.iloc[i].fgHaralick.flatten())))

labels = training_data.breed

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.33, stratify=labels, random_state=42)

print('[INFO] fitting model')
model = RandomForestClassifier(max_depth=50, n_estimators=40, random_state=42, n_jobs=-1) # best parameters
model.fit(X_train, y_train)

print('[INFO] baseline score:')
get_score(model, X_test)

# check what adding noise to the training data does
np.random.seed(42)
for pct in [0.1, 0.2, 0.3, 0.4]:
    X_train_noisy = X_train[:] # copy list
    for i in range(len(X_train_noisy)):
        X_train_noisy[i] = add_noise(X_train_noisy[i], pctNoise=pct)
    
    model = RandomForestClassifier(max_depth=50, n_estimators=40, random_state=42, n_jobs=-1) # best parameters
    model.fit(X_train_noisy, y_train)
    
    print('score with {}% noise in training data:'.format(pct*100))
    get_score(model, X_test)

for pct in [0.1, 0.2, 0.3, 0.4]:
    X_test_noisy = X_test[:] # copy list
    for i in range(len(X_test_noisy)):
        X_test_noisy[i] = add_noise(X_test_noisy[i], pctNoise=pct)
    
    model = RandomForestClassifier(max_depth=50, n_estimators=40, random_state=42, n_jobs=-1) # best parameters
    model.fit(X_train, y_train)
    
    print('score with {}% noise in testing data:'.format(pct*100))
    get_score(model, X_test_noisy)