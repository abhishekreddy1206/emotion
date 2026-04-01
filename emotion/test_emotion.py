#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script: loads a trained CNN model and runs predictions on test images.

@author: mary akshara allam
"""

import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

import tensorflow as tf
from tensorflow.keras.utils import to_categorical

np.random.seed(42)
DATA = "CK+48"
base_path = "CK+48/test_images/"
model_path = "CK+48/"
categories = ["Happy", "Fear", "Sadness", "Surprise", "Anger"]
NUM_CLASSES = len(categories)
INPUT_SHAPE = (48, 48, 1)
class_count = {}

for dir_ in os.listdir(base_path):
    if not dir_.isupper():
        os.rename(base_path + dir_, base_path + dir_.title())
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
            except (cv2.error, ValueError, OSError) as e:
                print(f"Warning: skipping {f}: {e}")

        print(f"loaded {dir_} images to numpy arrays...")
        label += 1

img_arr = img_arr / 255.
img_label = to_categorical(img_label, NUM_CLASSES)

# Load trained model
model = tf.keras.models.load_model(model_path + f"cnn_{DATA}_{NUM_CLASSES}emo.h5")

# Run predictions per emotion category
fig_num = 1
for emotion_idx, emotion in enumerate(categories):
    if emotion not in label_to_text.values():
        continue

    # Find indices belonging to this emotion
    emotion_label = [k for k, v in label_to_text.items() if v == emotion][0]
    emotion_indices = np.where(img_label[:, emotion_label] == 1)[0]

    if len(emotion_indices) == 0:
        continue

    num_to_show = min(8, len(emotion_indices))
    sample_indices = np.random.choice(emotion_indices, size=num_to_show, replace=False)

    plt.figure(fig_num, (16, 1.5))
    fig_num += 1

    for plot_idx, img_idx in enumerate(sample_indices):
        sample_img = img_arr[img_idx, :, :, 0]
        pred = label_to_text[np.argmax(model.predict(sample_img.reshape(1, 48, 48, 1)), axis=1)[0]]
        ax = plt.subplot(1, num_to_show + 1, plot_idx + 1)
        ax.imshow(sample_img, cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"t:{emotion[:3]}, p:{pred[:3]}")
        plt.tight_layout()

plt.show()
