from __future__ import print_function
import pandas as pd
import pickle as pk
import numpy as np
import time
import random
import threading
import json
from sklearn.cross_validation import train_test_split

# load configuration
with open('../../config.json', 'rb') as f:
    config = json.load(f)

mainImPath = config['image_dir']
pDir = config['pickle_dir']

# load training data
training_data = pk.load(open(pDir + 'histNtext-fg+bg.pd.pk', 'rb'))

Tdata = training_data.drop('breed', 1)
data = []
for i in range(Tdata.shape[0]):
    # only use the first 3 components
    data.append([i for i in Tdata.iloc[i, :3]])

labels = training_data.breed

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.33, stratify=labels, random_state=42)

print('y_test size:', y_test.shape[0])
print('expected correct classification from random guessing:', y_test.shape[0]/252.0)

breeds = labels.unique().tolist()

def random_guesses():
    global scores, breeds
    randPreds = []
    for i in range(y_test.shape[0]):
        randPreds.append(random.choice(breeds))
    score = 0
    for i in range(y_test.shape[0]):
        if randPreds[i]==y_test.iloc[i]:
            score+=1
    scores.append(score)

threads = []
scores = []
start = time.time()
for i in range(1000):
    t = threading.Thread(target=random_guesses)
    t.daemon = True
    t.start()
    threads.append(t)

for each in threads:
    each.join()

print('took', time.time()-start, 'seconds with multithreading')
print('expected correct predictions from random guessing:', np.mean(scores)) # comes out around 2

start = time.time()
scores = []
for i in range(1000):
    randPreds = []
    for i in range(y_test.shape[0]):
        randPreds.append(random.choice(breeds))
    score = 0
    for i in range(y_test.shape[0]):
        if randPreds[i]==y_test.iloc[i]:
            score+=1
    scores.append(score)

print('took', time.time()-start, 'seconds without multithreading') # faster without multithreading...don't understand
print('expected correct predictions from random guessing:', np.mean(scores))