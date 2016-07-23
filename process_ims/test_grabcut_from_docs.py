import numpy as np
import cv2
from matplotlib import pyplot as plt
import pickle as pk
import pandas as pd

mainImPath = '/media/nate/Windows/github/IDmyDog/scrape-ims/images/'

bb = pk.load(open('pickle_files/pDogs-bounding-boxes-clean.pd.pk', 'rb'))
bb.dropna(inplace=True)

for i in range(bb.shape[0]):
    entry = bb.iloc[i]
    img = cv2.imread(entry.path)
    bods = entry.bodies
    if len(bods) == 1:
        bd = bods[0]
        print(bd)
        rect = (bd[1][0], bd[0][1], -bd[1][0] + bd[0][0], bd[1][1] - bd[0][1])
        print(rect)
        #cv2.rectangle(img, (bd[1][0], bd[0][1]), (bd[0][0], bd[1][1]), BLUE, 2)
        mask = np.zeros(img.shape[:2],np.uint8)

        bgdModel = np.zeros((1,65),np.float64)
        fgdModel = np.zeros((1,65),np.float64)

        #rect = (50,50,450,290)
        cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)

        mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
        img = img*mask2[:,:,np.newaxis]
        plt.imshow(img),plt.colorbar(),plt.show()