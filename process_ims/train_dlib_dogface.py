# TODO: match aspect ratios of boxes

from __future__ import print_function
import dlib
from skimage import io

import pandas as pd
import pickle as pk
import cv2
import os
import imutils
import re

mainImPath = '/media/nate/Windows/github/IDmyDog/scrape-ims/images/'

bb = pk.load(open('pickle_files/pDogs-bounding-boxes-clean.pd.pk', 'rb'))
bb.dropna(inplace=True)

images = []
boxes = []
aspects = []
areas = []

for i in range(bb.shape[0]):
    cropDir = mainImPath + bb.iloc[i].breed + '/cropped/'
    # crop out body from image
    bods = bb.iloc[i].bodies
    imPath = bb.iloc[i].path
    imName = bb.iloc[i].path.split('/')[-1]
    ext = re.search('\.\w', imName)
    if ext:
        imName = imName.split('.')[0]
    im = cropDir + 'heads/' + imName + '.jpg'
    
    try:
        heads = bb.iloc[i].heads
        if len(heads) == 1:
            head = heads[0]
            ys = sorted([head[0][1], head[1][1]])
            xs = sorted([head[0][0], head[1][0]])
            aspect_ratio = (xs[1]-xs[0])/(ys[1]-ys[0])
            aspects.append(aspect_ratio)
            area = (xs[1]-xs[0]) * (ys[1]-ys[0])
            areas.append(area)
            if aspect_ratio > 1.5:
                images.append(io.imread(im))
                bbox = [dlib.rectangle(left=int(xs[0]), top=int(ys[0]), right=int(xs[1]-xs[0]), bottom=int(ys[1]-ys[0]))]
                boxes.append(bbox)
    except:
        print('couldn\'t load image')
        print(im)
        #image = cv2.imread(im)
        
        #cv2.imshow('',im)
        #cv2.waitKey(0)
        continue
    
        
options = dlib.simple_object_detector_training_options()
detector = dlib.train_simple_object_detector(images, boxes, options)
detector.save('test_dlib_detector')
win = dlib.image_window()
win.set_image(detector)
dlib.hit_enter_to_continue()

for imPath in bb.path:
	# load the image and make predictions
	image = cv2.imread(imPath)
	boxes = detector(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
 
	# loop over the bounding boxes and draw them
	for b in boxes:
		(x, y, w, h) = (b.left(), b.top(), b.right(), b.bottom())
		cv2.rectangle(image, (x, y), (w, h), (0, 255, 0), 2)
 
	# show the image
	cv2.imshow("Image", image)
	cv2.waitKey(0)