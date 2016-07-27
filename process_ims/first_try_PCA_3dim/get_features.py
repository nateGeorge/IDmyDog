from __future__ import print_function
from sklearn.decomposition import PCA
import cv2
from mahotas.features import haralick

def get_avg_hara(im, rects):
    # returns the haralick texture averaged over all rectangles in an image
    if len(rects)==0:
        return None
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    hara = 0
    for r in rects:
        # slice images as: img[y0:y1, x0:x1]
        hara += haralick(im[r[0][1]:r[1][1], r[0][0]:r[1][0]]).mean(0)
    hara /= (len(rects))
    return hara

def get_features(im, pcafit):
    '''
    takes input image, im, and extracts features from it
    uses supplied pca fit, pcafit
    '''
    h, w = im.shape[:2]
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    hara = haralick(gray).mean(0)
    newHara = []
    for h in hara:
        newHara.append(np.log(abs(h)))
    
    # need to reshape to avoid error in future for pca xform
    newHara = np.array(newHara).reshape(1, -1)
    xformH = pcafit.transform(newHara)
    return xformH

# load pca fit
pcaFG = pk.load(open('pickle_files/pcaFG.pk', 'rb'))