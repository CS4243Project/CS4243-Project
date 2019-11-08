from utils import pyramid
from utils import sliding_window
import argparse
import time
import cv2
import os.path as osp
import numpy as np
from sklearn.svm import LinearSVC, SVC
from utils import load_image_gray
import pickle
from skimage import feature
import os

dataset_dir = 'datasets/JPEGImages/'

if __name__ == "__main__":
    #test_image_feats = bags_of_sifts(test_image_paths, vocab_filename)
    with open('finalized_model.pkl', 'rb') as f:
        model = pickle.load(f)

    for image in os.listdir('test_images'):
        print(image)
        image = cv2.imread(os.path.join('test_images', image))
        cut = cv2.resize(image, (50, 50))
        (H, hogImage) = feature.hog(cut, orientations=9, pixels_per_cell=(4, 4),
                                cells_per_block=(2, 2), transform_sqrt=True, block_norm="L2", visualize=True)
        test_image_feat = H.flatten()
        from skimage import exposure
        hogImage = exposure.rescale_intensity(hogImage, out_range=(0, 255))
        hogImage = hogImage.astype("uint8")

        # cv2.imshow('imgagga', hogImage)
        # cv2.waitKey()

        test_label = model.predict([test_image_feat])
        # print(test_label, y, x)
        if test_label == 'waldo':
            # print(y)
            # print(x)
            print('waldo_found')
            cv2.imshow('imgagga', cut)
            cv2.waitKey()
            cv2.imshow('imgagga', hogImage)
            cv2.waitKey()

        if test_label == 'wenda':
            # print(y)
            # print(x)
            print('wenda_found')
            cv2.imshow('imgagga', cut)
            cv2.waitKey()
            cv2.imshow('imgagga', hogImage)
            cv2.waitKey()

        if test_label == 'wizard':
            # print(y)
            # print(x)
            print('wizard_found')
            cv2.imshow('imgagga', cut)
            cv2.waitKey()
            cv2.imshow('imgagga', hogImage)
            cv2.waitKey()

        if test_label == 'None':
            # print(y)
            # print(x)
            print('None')
            cv2.imshow('imgagga', cut)
            cv2.waitKey()
            cv2.imshow('imgagga', hogImage)
            cv2.waitKey()
