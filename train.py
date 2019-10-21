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

def build_vocabulary(image_list, vocab_size):
    dim = 128  # length of the SIFT descriptors that you are going to compute.
    vocab = np.zeros((vocab_size, dim))
    total_SIFT_features = np.zeros((20 * len(image_list), dim))

    imags = {}
    train_image_set = set()

    for i in range(len(image_list)):
        image_info = image_list[i]
        train_image_set.add(image_info['index'])

    for index in train_image_set:
        image_path = 'datasets/JPEGImages/' + index + '.jpg'
        img = load_image_gray(image_path)
        imags[index] = img
        print(index + "FINISHIED")

    print("Arrive Here")
    for i in range(len(image_list)):
        image_info = image_list[i]
        img = imags[image_info['index']]
        img = img[int(image_info['ymin']):int(image_info['ymax']), int(image_info['xmin']):int(image_info['xmax'])]
        frames, descriptors = vlfeat.sift.dsift(img, step=4, fast=True)
        idx = np.random.randint(descriptors.shape[0], size=20)
        total_SIFT_features[i * 20:(i + 1) * 20] = descriptors[idx, :]

    vocab = vlfeat.kmeans.kmeans(total_SIFT_features, vocab_size)
    return vocab

def bags_of_sifts(image_list, vocab):
    dim = vocab.shape[0]
    feats = np.zeros(shape=(len(image_list), dim))
    imags = {}
    train_image_set = set()

    for i in range(len(image_list)):
        image_info = image_list[i]
        train_image_set.add(image_info['index'])

    for index in train_image_set:
        image_path = 'datasets/JPEGImages/' + index + '.jpg'
        img = load_image_gray(image_path)
        imags[index] = img
        print(index + "FINISHIED")

    print("Start Feats")
    for i in range(len(image_list)):
        image_info = image_list[i]
        img = imags[image_info['index']]
        img = img[int(image_info['ymin']):int(image_info['ymax']), int(image_info['xmin']):int(image_info['xmax'])]
        frames, descriptors = vlfeat.sift.dsift(img, step=4, fast=True)
        distance = cdist(descriptors, vocab)
        min_index = np.unravel_index(np.argmin(distance, axis=1), distance.shape)[1]
        histogram, bin_edges = np.histogram(min_index, bins=np.arange(dim+1))
        histogram = histogram / np.linalg.norm(histogram)

        feats[i] = histogram

    return feats

if __name__ == "__main__":
    train_lists_filename = 'datasets/ImageSets/train.txt'
    annotation_path = 'datasets/Annotations/'
    check_list = ['xmin', 'xmax', 'ymin', 'ymax']
    train_image_list = []
    with open(train_lists_filename, "r") as f:
        train_list = f.readlines()
    for train_index in train_list:
        specified_region = []
        train_index = train_index.strip()
        dom = xml.dom.minidom.parse(annotation_path+train_index+'.xml')
        root = dom.documentElement
        size = root.getElementsByTagName('size')[0]
        width = int(size.getElementsByTagName('width')[0].firstChild.nodeValue)
        height = int(size.getElementsByTagName('height')[0].firstChild.nodeValue)
        objs = root.getElementsByTagName('object')

        for obj in objs:
            name = obj.getElementsByTagName('name')[0].firstChild.nodeValue
            bndbox = obj.getElementsByTagName('bndbox')[0]
            key_dict = {}
            for check_coord in check_list:
                coord = int(bndbox.getElementsByTagName(check_coord)[0].firstChild.nodeValue)
                key_dict[check_coord] = coord
            specified_region.append(key_dict)
            key_dict['index'] = train_index
            key_dict['name'] = name
            train_image_list.append(key_dict)

        time = 10 # may need to find out how many NON per image do we need
        while time > 0:
            xmin = random.randint(0, width - 50)
            xmax = random.randint(xmin + 50, width)
            ymin = random.randint(0, height - 50)
            ymax = random.randint(ymin + 50, height)
            flag = True
            for region in specified_region:
                if not (region['xmax'] < xmin or region['ymax'] < ymin or xmax < region['xmin'] or ymax < region['ymin']) :
                    flag = False
                    break
            if flag:
                key_dict = {'index': train_index, 'name': 'NON', 'xmax': xmax, 'xmin': xmin, 'ymax': ymax, 'ymin':ymin}
                train_image_list.append(key_dict)
                time -= 1

    train_tag_labels = [image['name'] for image in train_image_list]
    vocab_filename = 'vocab.pkl'
    train_feats_filename = 'train_feats.pkl'
    train_tag_filename = 'train_tag.pkl'
    if not osp.isfile(vocab_filename):
        print('No existing visual word vocabulary found. Computing one from training images')
        vocab_size = 200  # Larger values will work better (to a point) but be slower to compute
        vocab = build_vocabulary(train_image_list, vocab_size)

        with open(vocab_filename, 'wb') as f:
            pickle.dump(vocab, f)
            print('{:s} saved'.format(vocab_filename))
    else:
        with open(vocab_filename, 'rb') as f:
            vocab = pickle.load(f)

    if not osp.isfile(train_feats_filename):
        print('No existing visual training set feats found. Computing one from training images')
        with open(train_tag_filename, 'wb') as f:
            pickle.dump(train_tag_labels, f)
            print('{:s} saved'.format(train_tag_filename))

        train_image_feats = bags_of_sifts(train_image_list, vocab)
        with open(train_feats_filename, 'wb') as f:
            pickle.dump(train_image_feats, f)
            print('{:s} saved'.format(train_feats_filename))


