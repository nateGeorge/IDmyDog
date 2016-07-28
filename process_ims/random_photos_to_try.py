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