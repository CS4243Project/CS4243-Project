# from utils import pyramid
from utils import sliding_window
import argparse
import time
import cv2
import numpy as np
from sklearn.svm import LinearSVC, SVC
from utils import load_image_gray
import pickle


from skimage import exposure
from skimage import feature
import os

train_data_dir = 'trainingdata'


def build_bag_of_hog(image_list):

    print("Start Feats")
    feats = []
    # print(len(image_dict))
    for image_path in image_list:
        # print(image)
        # image_path = 'datasets/JPEGImages/' + image['index'] + '.jpg'
        img = load_image_gray(image_path)
        # img = cv2.imread(image_path)
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # images[index] = img

        # img = img[int(image['ymin']):int(image['ymax']), int(image['xmin']):int(image['xmax'])]
        # if image['name'] == 'waldo' or image['name'] == 'wenda' or image['name'] == 'wizard':
        #     cv2.imshow('imjsdkasg', img)
        #     cv2.waitKey()
        img = cv2.resize(img, (50, 50), interpolation = cv2.INTER_AREA)

        descriptor = feature.hog(img, orientations=9, pixels_per_cell=(4, 4),
                                    cells_per_block=(2, 2), transform_sqrt=True, block_norm="L2")
        descriptor = descriptor.flatten()
        feats.append(descriptor)
        print(image_path + ' ' + 'Done')

    return feats


if __name__ == "__main__":
    train_image_list = []
    train_image_labels = []
    for image in os.listdir(train_data_dir):
        print(image)
        train_image_list.append(os.path.join(train_data_dir, image))
        tokens = image.split('_')
        label = tokens[0]
        if label == 'wendy':
            label = 'wenda'
        if label == 'neg':
            label = 'None'
        train_image_labels.append(label)

    features = build_bag_of_hog(train_image_list)
    features = np.array(features)
    print(features.shape)
    clf = SVC(kernel='rbf', gamma='scale', decision_function_shape='ovc', probability=True)
    clf.fit(features, train_image_labels)
    print(type(clf))
    filename = 'finalized_model.pkl'
    pickle.dump(clf, open(filename, 'wb'))
    print('model saved!')


