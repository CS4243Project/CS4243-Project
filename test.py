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
import cyvlfeat as vlfeat
import pickle
import math
import xml.dom.minidom
import random

def bags_of_sifts(img, vocab):
    dim = vocab.shape[0]
    feats = np.zeros(shape=(1, dim))
    frames, descriptors = vlfeat.sift.dsift(img, step=4, fast=True)
    distance = cdist(descriptors, vocab)
    min_index = np.unravel_index(np.argmin(distance, axis=1), distance.shape)[1]
    histogram, bin_edges = np.histogram(min_index, bins=np.arange(dim + 1))
    histogram = histogram / np.linalg.norm(histogram)
    feats[0] = histogram
    return feats

def svm_classify(train_image_feats, train_labels, test_image_feats):
    categories = list(set(train_labels))
    test_labels = []

    clf = SVC(kernel='rbf', gamma='scale', decision_function_shape='ovc')
    clf.fit(train_image_feats, train_labels)
    test_labels = clf.predict(test_image_feats)

    return test_labels
if __name__ == "__main__":
    with open('vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)
    with open('train_feats.pkl', 'rb') as f:
        train_image_feats = pickle.load(f)
    with open('train_tag.pkl', 'rb') as f:
        train_tag = pickle.load(f)
    #test_image_feats = bags_of_sifts(test_image_paths, vocab_filename)
    image = load_image_gray('datasets/JPEGImages/003.jpg')
    (winW, winH) = (16, 16)
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
            test_image_feat = bags_of_sifts(cut, vocab)
            test_label = svm_classify(train_image_feats, train_tag, test_image_feat)
            print(test_label, y, x)
