import pandas as pd
import pickle as pk
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn.grid_search import GridSearchCV

# load training data
training_data = pk.load(open('pickle_files/fgHara-full-13x4.pd.pk', 'rb'))

data = []
for i in range(training_data.shape[0]):
    # only use the first 3 components
    data.append(training_data.iloc[i].fgHaralick.flatten())

labels = training_data.breed

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.33, stratify=labels, random_state=42)

clf = SVC(random_state=42)
params = {
          'kernel': ['rbf', 'sigmoid'],
          'C': [0.1, 1, 10.0],
          'gamma': ['auto', 1/2., 1/10., 1/26.]
         }

model = GridSearchCV(clf, params, cv=3)
model.fit(data, labels)
print("[INFO] best hyperparameters: {}".format(model.best_params_))
# best parameters so far were found to be
model.fit(X_train, y_train)

model.score(X_test, y_test)

predictions = model.predict(X_test)

score = 0
for i in range(predictions.shape[0]):
    if predictions[i]==y_test.iloc[i]:
        score+=1

print(score)

#print(classification_report(y_test, predictions))

mainImPath = '/media/nate/Windows/github/IDmyDog/scrape-ims/images/'
testIm = cv2.imread(mainImPath + 'Appenzeller Sennenhunde/' + 'Appenzeller Sennenhunde1-4.jpeg')
testIm2 = cv2.imread(mainImPath + 'German Shepherd Dog/' + 'German Shepherd Dog1-5.jpeg')
testIm3 = cv2.imread(mainImPath + 'German Shepherd Dog/' + '8ae977feb85f255202ea1c8dba3d331fd29a188c.jpg')
testIm4 = cv2.imread(mainImPath + 'German Shepherd Dog/' + 'German Shepherd Dog1-0.jpeg')
testIm2cropped = testIm2[150:200, 200:300]

feats = get_features(testIm, pcaFG)
feats2 = get_features(testIm2, pcaFG)
feats3 = get_features(testIm3, pcaFG)
feats4 = get_features(testIm3, pcaFG)
feats2cr = get_features(testIm2cropped, pcaFG)

model.predict(feats[0][:3].reshape(1, -1)) # should be appenzeller
model.predict(feats2[0][:3].reshape(1, -1)) # should be german shephard
model.predict(feats3[0][:3].reshape(1, -1)) # should be german shephard
model.predict(feats4[0][:3].reshape(1, -1)) # should be german shephard
model.predict(feats2cr[0][:3].reshape(1, -1)) # should be german shephard