# I used python 2.7 for this file

from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
import json
import cPickle as pk
import os
import numpy as np
import pandas as pd
from time import time

def load_everything():
    # load configuration
    with open('../config.json', 'rb') as f:
        config = json.load(f)

    mainImPath = config['image_dir']
    pDir = config['pickle_dir']

    bb = pk.load(open(pDir + 'pDogs-bounding-boxes-clean.pd.pk', 'rb'))
    bb.dropna(inplace=True)

    breeds = list(sorted(bb.breed.unique().tolist()))

    return breeds, bb, mainImPath

def create_model():
    base_model = VGG19(weights='imagenet')
    model = Model(input=base_model.input, output=base_model.get_layer('block4_pool').output)
    return model

def extract_feats(img_path):
        try:
            img = image.load_img(img_path, target_size=(224, 224))
        except IOError:
            print 'couldn\'t load file'
            return None

        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        block4_pool_features = model.predict(x)
        return block4_pool_features

def extract_all_feats(timing=True):
    all_feats = None
    for b in breeds:
        print 'on breed:', b
        imDir = mainImPath + breeds[0] + '/'
        ims = os.listdir(imDir)
        for im in ims:
            start = time()
            print 'on image:', im
            b4 = extract_feats(imDir + im)
            if b4 is None:
                continue

            if all_feats is None:
                files = [im]
                all_feats = b4
            else:
                files.append(im)
                all_feats = np.c_[all_feats, all_feats]

            if timing:
                print 'took', time() - start, 'seconds'

    return files, all_feats

def test(timing=True):
    start = time()
    testImDir = mainImPath + breeds[0] + '/'
    testIms = os.listdir(testImDir)
    print 'extracting features for', breeds[0]
    print 'file:', testIms[0]
    b4 = extract_feats(testImDir + testIms[0])
    if timing:
        runtime = time() - start
        print 'took', runtime, 'seconds'

    return b4, runtime

if __name__ == "__main__":
    print 'loading data'
    breeds, bb, mainImPath = load_everything()
    model = create_model()
    runtest = False
    if runtest:
        testFeat, runtime = test()
        print 'array size (bytes):', testFeat.nbytes
        print 'total expected size:', (testFeat.nbytes * bb.shape[0]) / 1000000000., 'GB'
        print 'total run time:', runtime * bb.shape[0] / 60, 'minutes'

    files_all_feats = extract_all_feats()
    pk.dump(files_all_feats, open('files_all_feats.pk', w), 2)
