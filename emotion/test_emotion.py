#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: mary akshara allam
"""

import os
import cv2
import math
import time

import numpy as np
import pandas as pd

import scikitplot, keras
import seaborn as sns
from matplotlib import pyplot as plt
import skimage.io, skimage, skimage.feature


import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.utils import plot_model
from keras.utils import np_utils

np.random.seed(42)
DATA = "CK+48"
base_path = "CK+48/test_images/"
categories = ["Happy", "Fear", "Sadness", "Surprise", "Anger"]
NUM_CLASSES = len(categories)
class_count = {}
img_width, img_height = 224, 224
INPUT_SHAPE = (48,48,1)


for dir_ in os.listdir(base_path):
    if not dir_.isupper():
        os.rename(base_path+dir_, base_path+dir_.title())
        dir_ = dir_.title()

    count = 0
    for f in os.listdir(base_path + dir_ + "/"):
        count += 1

    class_count[dir_] = count
    print(f"{dir_} has {count} number of images")

total_images = sum(class_count.values())
img_arr = np.empty(shape=(total_images, *INPUT_SHAPE))
img_label = np.empty(shape=(total_images))
label_to_text = {}

i = 0
label = 0
for dir_ in os.listdir(base_path):

    if dir_ in categories:
        label_to_text[label] = dir_
        for f in os.listdir(base_path + dir_ + "/"):
            try:
                img_arr[i] = np.expand_dims(cv2.imread(base_path + dir_ + "/" + f, 0), axis=2)
                img_label[i] = label
                i += 1
            except:
                pass

        print(f"loaded {dir_} images to numpy arrays...")
        label += 1

img_label = np_utils.to_categorical(img_label)
img_arr.shape, img_label.shape
model = keras.models.load_model(base_path+f"cnn_{DATA}_{NUM_CLASSES}emo.h5")
f = 1
for emotion in categories:
    plt.figure(f, (16,1.5))
    f += 1

    for i,img_idx in enumerate(img_arr):
        sample_img = img_arr[i]
        pred = label_to_text[np.argmax(model.predict(sample_img.reshape(1,48,48,1)), axis=1)[0]]
        ax = plt.subplot(1, 9, i+1)
        ax.imshow(sample_img, cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"t:{emotion[:3]}, p:{pred[:3]}")
        plt.tight_layout()
