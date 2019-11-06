# from utils import pyramid
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

from skimage import exposure
from skimage import feature
import os

train_data_dir = 'trainingdata'


# def build_bag_of_hog(image_dict):
#
#     print("Start Feats")
#     feats = []
#     print(len(image_dict))
#     for image in image_dict:
#         # print(image)
#         image_path = 'datasets/JPEGImages/' + image['index'] + '.jpg'
#         img = load_image_gray(image_path)
#         # img = cv2.imread(image_path)
#         # img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#         # images[index] = img
#
#         img = img[int(image['ymin']):int(image['ymax']), int(image['xmin']):int(image['xmax'])]
#         if image['name'] == 'waldo' or image['name'] == 'wenda' or image['name'] == 'wizard':
#             cv2.imshow('imjsdkasg', img)
#             cv2.waitKey()
#         img = cv2.resize(img, (32, 32), interpolation = cv2.INTER_AREA)
#
#         descriptor = feature.hog(img, orientations=9, pixels_per_cell=(4, 4),
#                                     cells_per_block=(2, 2), transform_sqrt=True, block_norm="L1")
#         descriptor = descriptor.flatten()
#         feats.append(descriptor)
#         print(image['index'] + ' ' + 'Done')
#
#     return feats

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
                                    cells_per_block=(2, 2), transform_sqrt=True, block_norm="L1")
        descriptor = descriptor.flatten()
        feats.append(descriptor)
        print(image_path + ' ' + 'Done')

    return feats


if __name__ == "__main__":
    # train_lists_filename = 'datasets/ImageSets/train.txt'
    # annotation_path = 'datasets/Annotations/'
    # check_list = ['xmin', 'xmax', 'ymin', 'ymax']
    # train_image_list = []
    # with open(train_lists_filename, "r") as f:
    #     train_list = f.readlines()
    # for train_index in train_list:
    #     specified_region = []
    #     train_index = train_index.strip()
    #     dom = xml.dom.minidom.parse(annotation_path+train_index+'.xml')
    #     root = dom.documentElement
    #     size = root.getElementsByTagName('size')[0]
    #     width = int(size.getElementsByTagName('width')[0].firstChild.nodeValue)
    #     height = int(size.getElementsByTagName('height')[0].firstChild.nodeValue)
    #     objs = root.getElementsByTagName('object')
    #
    #     for obj in objs:
    #         name = obj.getElementsByTagName('name')[0].firstChild.nodeValue
    #         bndbox = obj.getElementsByTagName('bndbox')[0]
    #         key_dict = {}
    #         for check_coord in check_list:
    #             coord = int(bndbox.getElementsByTagName(check_coord)[0].firstChild.nodeValue)
    #             key_dict[check_coord] = coord
    #         specified_region.append(key_dict)
    #         key_dict['index'] = train_index
    #         key_dict['name'] = name
    #         train_image_list.append(key_dict)
    #
    #     time = 10 # may need to find out how many NON per image do we need
    #     while time > 0:
    #         xmin = random.randint(0, width - 50)
    #         xmax = random.randint(xmin + 50, width)
    #         ymin = random.randint(0, height - 50)
    #         ymax = random.randint(ymin + 50, height)
    #         flag = True
    #         for region in specified_region:
    #             if not (region['xmax'] < xmin or region['ymax'] < ymin or xmax < region['xmin'] or ymax < region['ymin']) :
    #                 flag = False
    #                 break
    #         if flag:
    #             key_dict = {'index': train_index, 'name': 'NON', 'xmax': xmax, 'xmin': xmin, 'ymax': ymax, 'ymin':ymin}
    #             train_image_list.append(key_dict)
    #             time -=
    train_image_list = []
    train_image_labels = []
    for image in os.listdir(train_data_dir):
        print(image)
        train_image_list.append(os.path.join(train_data_dir, image))
        tokens = image.split('_')
        label = tokens[0]
        if label == 'wendy':
            label = 'wenda'
        train_image_labels.append(label)

    print(train_image_list)
    print(train_image_labels)



    # train_tag_labels = [image['name'] for image in train_image_list]
    # print(len(train_image_list))
    vocab_filename = 'vocab.pkl'
    train_feats_filename = 'train_feats.pkl'
    train_tag_filename = 'train_tag.pkl'
    images = {}
    # train_image_set = set()
    # for image_dict in train_image_list:
    #     if image_dict['name'] == 'waldo' or image_dict['name'] == 'wenda' or image_dict['name'] == 'wizard':
    # print(train_image_list)

    features = build_bag_of_hog(train_image_list)
    features = np.array(features)
    print(features.shape)
    clf = SVC(kernel='rbf', gamma='scale', decision_function_shape='ovc')
    clf.fit(features, train_image_labels)
    print(type(clf))
    filename = 'finalized_model.pkl'
    pickle.dump(clf, open(filename, 'wb'))
    print('model saved!')


