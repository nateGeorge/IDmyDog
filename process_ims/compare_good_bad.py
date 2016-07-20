import seaborn as sns
import numpy as np
import cv2
import os
import pandas as pd
import pickle
import mahotas as mh
import matplotlib.pyplot as plt
plt.style.use('seaborn-dark')

# load hand-picked good images in 'train' folders 
try:
    breed_ims_good = pickle.load(open('pickle_files/breed_ims-aff-afg-good.pd.pk', 'rb'))
except IOError:
    imPath = '/media/nate/Windows/github/IDmyDog/scrape-ims/images/'
    breeds = os.listdir(imPath)[:2] # only use the first two breeds for now
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
    
    breed_ims_good = pd.DataFrame(pdDict)
    
    pickle.dump(breed_ims_good, open('pickle_files/breed_ims-aff-afg-good.pd.pk', 'wb'))

# load hand-picked bad images in 'complex' folders
try:
    breed_ims_bad = pickle.load(open('pickle_files/breed_ims-aff-afg-bad.pd.pk', 'rb'))
except IOError:
    imPath = '/media/nate/Windows/github/IDmyDog/scrape-ims/images/'
    breeds = os.listdir(imPath)[:2] # only use the first two breeds for now
    print(breeds)   
    
    pdDict = {}
    pdDict['picPaths'] = []
    pdDict['images'] = []
    pdDict['breed_labels'] = []
    for breed in breeds:
        breedFolder = imPath + breed
        pics = os.listdir(breedFolder + '/complex')
        for pic in pics:
            picPath = breedFolder + '/' + pic
            print(breed, pic)
            image = cv2.imread(picPath)
            if image!=None: # some image links redirected to a 404, etc
                pdDict['picPaths'].append(picPath)
                pdDict['images'].append(cv2.imread(picPath))
                pdDict['breed_labels'].append(breed)
    
    breed_ims_bad = pd.DataFrame(pdDict)
    
    pickle.dump(breed_ims_bad, open('pickle_files/breed_ims-aff-afg-bad.pd.pk', 'wb'))

def plot_good_n_bad():
    # get example color histograms of good and bad images
    
    # GOOD #1
    f, axarr = plt.subplots(2, 2)
    
    chans = cv2.split(breed_ims_good.images[0])
    colors = ("b", "g", "r")
    axarr[0,0].set_title("Color Histogram of good image 1")
    #axarr[0,0].set_xlabel("Bins")
    axarr[0,0].set_ylabel("Normalized # of Pixels")
    stdDevs = []
    # loop over the image channels
    for (chan, color) in zip(chans, colors):
        # create a histogram for the current channel and plot it
        hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
        hist /= hist.sum() # normalize to compare images of different sizes
        stdDevs.append(hist.std())
        axarr[0,0].plot(hist, color = color)
        axarr[0,0].set_xlim([0, 256])
    
    print('stdev of good img 1:', stdDevs, np.average(stdDevs))
    
    # GOOD #2
    chans = cv2.split(breed_ims_good.images[1])
    axarr[0,1].set_title("Color Histogram of good image 2")
    #axarr[0,1].set_xlabel("Bins")
    #axarr[0,1].set_ylabel("# of Pixels")
    stdDevs = []
    # loop over the image channels
    for (chan, color) in zip(chans, colors):
        # create a histogram for the current channel and plot it
        hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
        hist /= hist.sum() # normalize to compare images of different sizes
        stdDevs.append(hist.std())
        axarr[0,1].plot(hist, color = color)
        axarr[0,1].set_xlim([0, 256])
    
    print('stdev of good img 2:', stdDevs, np.average(stdDevs))
    
    # BAD #1
    chans = cv2.split(breed_ims_bad.images[0])
    axarr[1,0].set_title("'Flattened' Color Histogram of bad image 1")
    axarr[1,0].set_xlabel("Bins")
    axarr[1,0].set_ylabel("Normalized # of Pixels")
    stdDevs = []
    # loop over the image channels
    for (chan, color) in zip(chans, colors):
        # create a histogram for the current channel and plot it
        hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
        hist /= hist.sum() # normalize to compare images of different sizes
        stdDevs.append(hist.std())
        axarr[1,0].plot(hist, color = color)
        axarr[1,0].set_xlim([0, 256])
    
    print('stdev of bad img 1:', stdDevs, np.average(stdDevs))
    
    # BAD #2
    chans = cv2.split(breed_ims_bad.images[1])
    axarr[1,1].set_title("'Flattened' Color Histogram of bad image 2")
    axarr[1,1].set_xlabel("Bins")
    #axarr[1,1].set_ylabel("# of Pixels")
    stdDevs = []
    # loop over the image channels
    for (chan, color) in zip(chans, colors):
        # create a histogram for the current channel and plot it
        hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
        hist /= hist.sum() # normalize to compare images of different sizes
        stdDevs.append(hist.std())
        axarr[1,1].plot(hist, color = color)
        axarr[1,1].set_xlim([0, 256]) 
    
    print('stdev of bad img 2:', stdDevs, np.average(stdDevs))
    
    plt.show()

def show_good_n_bad():
    # show example images used to generate color histograms
    f, axarr = plt.subplots(2, 2)
    
    axarr[0,0].imshow(breed_ims_good.images[0])
    axarr[0,1].imshow(breed_ims_good.images[1])
    axarr[1,0].imshow(breed_ims_bad.images[0])
    axarr[1,1].imshow(breed_ims_bad.images[1])
    axarr[0,0].set_title('Good image 1')
    axarr[0,1].set_title('Good image 2')
    axarr[1,0].set_title('Bad image 1')
    axarr[1,1].set_title('Bad image 2')
    plt.show()

def get_color_stds(dataFr):
    # returns DF of standard deviation of color histograms
    # takes a pandas dataframe with 'images' column
    # images should be loaded using cv2.imread()
    stds = []
    for im in dataFr.images:
        chans = cv2.split(im)
        tmpStd = []
        for chan in chans:
            hist = cv2.calcHist([chans[0]], [0], None, [256], [0, 256])
            hist /= hist.sum() # normalize to compare images of different sizes
            tmpStd.append(hist.std())
        stds.append(tmpStd + [np.average(tmpStd)])
    
    return pd.DataFrame(stds, columns = ['b', 'g', 'r', 'avg'])

good_stds = get_color_stds(breed_ims_good)
bad_stds = get_color_stds(breed_ims_bad)