import seaborn as sns
import numpy as np
import cv2
import os
import pandas as pd
import pickle
import pylab as plt
import mahotas as mh
from sklearn.svm import LinearSVC, SVC

if not os.path.exists('pickle_files'):
    os.makedirs('pickle_files')

# load images from pickle otherwise 
try:
    breed_ims = pickle.load(open('pickle_files/breed_ims-aff-afg-test.pd.pk', 'rb'))
except IOError:
    imPath = '/media/nate/Windows/github/IDmyDog/scrape-ims/images/'
    breeds = os.listdir(imPath)[:2]
    if 'full' in breeds:
        breeds.remove('full')
    
    print(breeds)   
    
    pdDict = {}
    pdDict['picPaths'] = []
    pdDict['images'] = []
    pdDict['breed_labels'] = []
    for breed in breeds:
        breedFolder = imPath + breed
        pics = os.listdir(breedFolder)
        for pic in pics:
            picPath = breedFolder + '/' + pic
            print(breed, pic)
            image = cv2.imread(picPath)
            if image!=None: # some image links redirected to a 404, etc
                pdDict['picPaths'].append(picPath)
                pdDict['images'].append(cv2.imread(picPath))
                pdDict['breed_labels'].append(breed)
    
    breed_ims = pd.DataFrame(pdDict)
    
    pickle.dump(breed_ims, open('pickle_files/breed_ims-aff-afg-test.pd.pk', 'wb'))

# exmaine height/width distribution
Hs = []
Ws = []
minH = np.inf
minW = np.inf
dims = []
for i in range(breed_ims.images.shape[0]):
    try:
        h, w, _ = breed_ims.images[i].shape
        Hs.append(h)
        Ws.append(w)
        dims.append([h, w])
        if h < minH:
            minH = h
        if w < minW:
            minW = w
    except AttributeError as e:
        print(e)
        print(breed_ims.picPaths[i])

dims = pd.DataFrame(dims, columns = ['height', 'width'])
g = sns.JointGrid("height", "width", dims)#, xlim=(-6,6), ylim=(-5,5))
g = g.plot_joint(sns.kdeplot, cmap="Blues", shade=True)
g = g.plot_marginals(sns.kdeplot, shade=True)
plt.show()

# widths are always larger than heights--expected

sns.lmplot("height", "width", dims, fit_reg=False)
plt.show()

# we can see there are a few tiny images, so we throw away any that have a dimension
# less than 250
for i in range(dims.shape[0]):
    h,w = dims.iloc[i]
    if h<250 or w<250:
        breed_ims.drop(breed_ims.index[[i]], inplace=True)

# compute haralick texture features
data = []
labels = []
for i in range(breed_ims.shape[0]):
    im = breed_ims['images'].iloc[i]
    image = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    features = mh.features.haralick(image).mean(axis=0)
    data.append(features)
    labels.append(breed_ims['breed_labels'].iloc[i])

data = np.array(data)
labels = np.array(labels)
dataLab = np.hstack((data, labels[:,np.newaxis]))
haralick = pd.DataFrame(dataLab, columns=['feature_{}'.format(i) for i in range(len(data[0]))] + ['breed'])

# convert to floats for plotting compatibility
for i in range(len(data[0])):
     haralick['feature_{}'.format(i)] = haralick['feature_{}'.format(i)].astype('float64')

sns.lmplot("feature_2", "feature_3", haralick, fit_reg=False)
plt.show()

sns.pairplot(haralick[['feature_{}'.format(i) for i in range(7)] + ['breed']], hue='breed')
plt.show()

sns.pairplot(haralick[['feature_{}'.format(i) for i in range(7,13)] + ['breed']], hue='breed')
plt.show()

model = SVC(random_state=42)
model.fit(data, labels)

def extract_haralick(images):
    data = []
    for im in images:
        image = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        features = mh.features.haralick(image).mean(axis=0)
        data.append(features)       
    
    return data


testim = cv2.imread('../scrape-ims/images/test/afghan hound/Afghan-Hound-4.jpg')
graytest = cv2.cvtColor(testim, cv2.COLOR_BGR2GRAY)

test_d = extract_haralick([testim])

print(model.predict(test_d)) # it fails with LinearSVC, but works with regular SVC with default params
# for most images, we need better segmentation of the dog from the background

# extract HOG features
(H, hogImage) = feature.hog(graytest, orientations=9, pixels_per_cell=(8, 8)
                            cells_per_block=(2, 2), transform_sqrt=True, visualise=True)
hogImage = exposure.rescale_intensity(hogImage, out_range=(0, 255))
hogImage = hogImage.astype("uint8")
plt.imshow(hogImage)
plt.show()


# try again, only using hand-picked images in 'train' folders 
try:
    breed_ims = pickle.load(open('pickle_files/breed_ims-aff-afg-test2.pd.pk', 'rb'))
except IOError:
    imPath = '/media/nate/Windows/github/IDmyDog/scrape-ims/images/'
    breeds = os.listdir(imPath)[:2]
    print(breeds)   
    
    pdDict = {}
    pdDict['picPaths'] = []
    pdDict['images'] = []
    pdDict['breed_labels'] = []
    for breed in breeds:
        breedFolder = imPath + breed
        pics = os.listdir(breedFolder + '/train')
        for pic in pics:
            picPath = breedFolder + '/' + pic
            print(breed, pic)
            image = cv2.imread(picPath)
            if image!=None: # some image links redirected to a 404, etc
                pdDict['picPaths'].append(picPath)
                pdDict['images'].append(cv2.imread(picPath))
                pdDict['breed_labels'].append(breed)
    
    breed_ims = pd.DataFrame(pdDict)
    
    pickle.dump(breed_ims, open('pickle_files/breed_ims-aff-afg-test2.pd.pk', 'wb'))





# if not done downloading images, only want breeds with many ims
breed_data = []
breed_target = []

imsCnt = breed_ims['breed_labels'].value_counts()
highIms = [imsCnt.index[i] for i in range(len(imsCnt)) if imsCnt[i] > 10]
for i in range(breed_ims.images.shape[0]):
    # todo; try except for nonetype object
    if breed_ims['breed_labels'][i] in highIms:
        breed_data.append(breed_ims.images[i].flatten())
        breed_target.append(breed_ims['breed_labels'][i])

# introspect the images arrays to find the shapes (for plotting)
n_samples, h, w = breed_ims.images.shape
np.random.seed(42)




Hs = []
Ws = []

for i in range(breed_ims.images.shape[0]):
    try:
        h, w, _ = breed_ims.images[i].shape
        Hs.append(h)
        Ws.append(w)
        if h < minH:
            minH = h
        if w < minW:
            minW = w
    except AttributeError as e:
        print(e)
        print(breed_ims.picPaths[i])
        
sns.distplot(Ws)
plt.show()
sns.distplot(Hs)
plt.show()