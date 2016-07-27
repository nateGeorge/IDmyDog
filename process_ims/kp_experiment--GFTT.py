# experimenting with fast hessian keypoint detection and SIFT feature extraction
from __future__ import print_function
import pandas as pd
import pickle as pk
import cv2
import os
import imutils
import numpy as np

def rootSift(image, kps, eps=1e-7):
    global detector, extractor
    # compute SIFT descriptors
    (kps, descs) = extractor.compute(image, kps)

    # if there are no keypoints or descriptors, return an empty tuple
    if len(kps) == 0:
        return ([], None)

    # apply the Hellinger kernel by first L1-normalizing and taking the
    # square-root
    descs /= (descs.sum(axis=1, keepdims=True) + eps)
    descs = np.sqrt(descs)

    # return a tuple of the keypoints and descriptors
    return (kps, descs)

mainImPath = '/media/nate/Windows/github/IDmyDog/scrape-ims/images/'



bb = pk.load(open('pickle_files/pDogs-bounding-boxes-clean.pd.pk', 'rb'))
bb.dropna(inplace=True)

detector = cv2.FeatureDetector_create("GFTT")
extractor = cv2.DescriptorExtractor_create("SIFT")

for i in range(bb.shape[0]):
    # crop out body from image
    bods = bb.iloc[i].bodies
    imPath = bb.iloc[i].path
    image = cv2.imread(imPath)
    cv2.imshow('original', image)
    for body in bods:
        ys = sorted([body[0][1], body[1][1]])
        xs = sorted([body[0][0], body[1][0]])
        crBod = image[ys[0]:ys[1], xs[0]:xs[1]]
        orig = crBod.copy()
        gray = cv2.cvtColor(crBod, cv2.COLOR_BGR2GRAY)
        kps = detector.detect(gray)
        print("# of keypoints: {}".format(len(kps)))
        kps, descs = rootSift(gray, kps)
        # loop over the keypoints and draw them
        for kp in kps:
            r = int(0.5 * kp.size)
            (x, y) = np.int0(kp.pt)
            cv2.circle(crBod, (x, y), r, (0, 255, 255), 2)
        cv2.imshow('', np.hstack([orig, crBod]))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    '''
    # crop out heads
    heads = bb.iloc[i].heads
    for head in heads:
        ys = sorted([head[0][1], head[1][1]])
        xs = sorted([head[0][0], head[1][0]])
        crHead = image[ys[0]:ys[1], xs[0]:xs[1]]
        orig = crHead.copy()
        gray = cv2.cvtColor(crHead, cv2.COLOR_BGR2GRAY)
        kps = detector.detect(gray)
        print("# of keypoints: {}".format(len(kps)))
        kps, descs = rootSift(gray, kps)
        # loop over the keypoints and draw them
        for kp in kps:
            r = int(0.5 * kp.size)
            (x, y) = np.int0(kp.pt)
            cv2.circle(crHead, (x, y), r, (0, 255, 255), 2)
        cv2.imshow('', np.hstack([orig, crHead]))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    '''