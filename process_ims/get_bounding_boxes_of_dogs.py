from __future__ import print_function
import cv2
import os
import numpy as np
import pandas as pd
import random

pDogs = pd.DataFrame(columns=['breed','path','bodies','heads'])
image = None
imPath = '/media/nate/Windows/github/IDmyDog/scrape-ims/images/'
breeds = os.listdir(imPath)
breeds.remove('full')
for breed in breeds:
    breedFolder = imPath + breed
    pics = os.listdir(breedFolder)
    for pic in pics:
        if os.path.isfile(breedFolder + '/' + pic):
            pDogs.append({'path':pic, 'breed':breedFolder + '/'}, ignore_index=True)
            '''if image == None:
                image = cv2.imread(breedFolder + '/' + pic)
                clone = image.copy()*/'''

refPt = []
def getBBs(event, x, y, flags, param):
    # takes BGR CV2 image as an input
    # displays image and waits for
    # bounding boxes to be clicked
    # around dogs' bodies and faces
    # returns np array of bounding boxes
    global pdDict, refPt
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [(x, y)]
        pdList[0].append([(x, y)])
        print('mouse down', x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates and indicate that
        # the cropping operation is finished
        refPt.append((x,y))
        pdList[0].append((x, y))
        print('mouse up', x, y)
        
        # draw a rectangle around the region of interest
        cv2.rectangle(image, refPt[0], refPt[1], (0, 255, 0), 2)
        cv2.imshow("image", image)

cv2.namedWindow("image")
cv2.setMouseCallback("image", getBBs)
 
# keep looping until the 'q' key is pressed
while True:
    # display the image and wait for a keypress
    cv2.imshow("image", image)
    key = cv2.waitKey(1) & 0xFF
    
    # if the 'r' key is pressed, reset the cropping region
    if key == ord("r"):
        image = clone.copy()
    
    # if the 'b' key is pressed, log position of dogs' bodies
    if key == ord('b'):
        bodies = True
    
    # if the 'f' key is pressed, log position of dogs' faces
    if key == ord('f'):
        faces = True
    
    # if the 'n' key is pressed, go to next dog pic
    if key == ord('n'):
        faces = True
        
    # if the 'd' key is pressed, go to next breed
    if key == ord('d'):
        faces = True
    
    # if the 'c' key is pressed, break from the loop
    elif key == ord("c"):
        break