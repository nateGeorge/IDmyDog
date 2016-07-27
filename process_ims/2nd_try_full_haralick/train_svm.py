import pandas as pd
import pickle as pk
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn.grid_search import GridSearchCV

# load training data
training_data = pk.load(open('../pickle_files/fgHDF-full.pd.pk', 'rb'))

Tdata = training_data.drop('breed', 1)
data = []
for i in range(Tdata.shape[0]):
    # only use the first 3 components
    data.append([i for i in Tdata.iloc[i, :3]])

labels = training_data.breed

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.33, stratify=labels, random_state=42)

clf = SVC(random_state=42)
params = {
          'kernel': ['rbf', 'sigmoid'],
          'C': [10.0, 15.0, 20.0, 30.0],
          'gamma': ['auto', 1, 1/2., 1/5., 1/10.]
         }

model = GridSearchCV(clf, params, cv=3)
model.fit(data, labels)
print("[INFO] best hyperparameters: {}".format(model.best_params_))
# best parameters so far were found to be
model = SVC(kernel='rbf', C=10.0, gamma=0.5, random_state=42)
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