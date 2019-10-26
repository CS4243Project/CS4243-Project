from utils import pyramid
from utils import sliding_window
import argparse
import time
import cv2
import os.path as osp
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.svm import LinearSVC, SVC
from utils import load_image_gray
import pickle
import math
import xml.dom.minidom
import random
from skimage import feature


if __name__ == "__main__":
    #test_image_feats = bags_of_sifts(test_image_paths, vocab_filename)
    with open('finalized_model.pkl', 'rb') as f:
        model = pickle.load(f)
    # image = load_image_gray('datasets/JPEGImages/003.jpg')

    image = cv2.imread('datasets/JPEGImages/003.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # cv2.imshow('og', image)
    # cv2.waitKey(0)
    (winW, winH) = (16, 16)
    # hog = cv2.HOGDescriptor()
    # loop over the image pyramid
    for resized in pyramid(image, scale=1.5):
        # loop over the sliding window for each layer of the pyramid
        for (x, y, window) in sliding_window(resized, stepSize=32, windowSize=(winW, winH)):
            # if the window does not meet our desired window size, ignore it
            if window.shape[0] != winH or window.shape[1] != winW:
                continue

            # THIS IS WHERE YOU WOULD PROCESS YOUR WINDOW, SUCH AS APPLYING A
            # MACHINE LEARNING CLASSIFIER TO CLASSIFY THE CONTENTS OF THE
            # WINDOW

            # since we do not have a classifier, we'll just draw the window
            clone = resized.copy()
            cut = clone[y: y+winH, x: x+winW]
            cut = cv2.resize(cut, (32, 32))
            (H, hogImage) = feature.hog(cut, orientations=9, pixels_per_cell=(8, 8),
                                    cells_per_block=(2, 2), transform_sqrt=True, block_norm="L1", visualize=True)
            test_image_feat = H.flatten()
            # from skimage import exposure
            # hogImage = exposure.rescale_intensity(hogImage, out_range=(0, 255))
            # hogImage = hogImage.astype("uint8")

            test_label = model.predict([test_image_feat])
            # print(test_label, y, x)
            if test_label == 'waldo':
                print(y)
                print(x)
                cv2.imshow('imgagga', cut)
                cv2.waitKey()
                cv2.imshow('imgagga', hogImage)
                cv2.waitKey()
