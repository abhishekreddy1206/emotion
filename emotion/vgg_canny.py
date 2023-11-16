#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 20:29:10 2021

@author: abhishek
"""

import os
import numpy as np
import keras
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D , Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
import skimage.io, skimage, skimage.feature
from skimage.color import rgb2gray, gray2rgb


def custom_img_prep_samp(img):
  img_2d = rgb2gray(img)
  canny_edges = skimage.feature.canny(img_2d)
  canny_edges_sigma2 = skimage.feature.canny(img_2d, sigma=2)
  canny_edges_sigma3 = skimage.feature.canny(img_2d, sigma=3)
  figure, (img1, img2, img3, img4) = pyplot.subplots(nrows=1, ncols=4, figsize=(8, 3), sharex=True, sharey=True)

  print(img.dtype)
  print(img_2d.dtype)
  print((gray2rgb(canny_edges)).dtype)
  img1.imshow(img, cmap=pyplot.cm.gray)
  img2.imshow(canny_edges, cmap=pyplot.cm.gray)
  img3.imshow(canny_edges_sigma2, cmap=pyplot.cm.gray)
  canny_edges_sigma2 = gray2rgb(img_2d)
  img4.imshow(canny_edges_sigma2, cmap=pyplot.cm.gray)

  figure.tight_layout()

  pyplot.show()

#data_dir = "CK+48/data/"
data_dir = "images/"
img = skimage.io.imread(data_dir + '/train/Anger/anger_1.jpg')
custom_img_prep_samp(img)


def custom_img_prep(img):
  img_2d = rgb2gray(img)
  canny_edges_sigma2 = skimage.feature.canny(img_2d, sigma=2)
  canny_edges_sigma2 = canny_edges_sigma2.astype(int)
  canny_edges_sigma2[canny_edges_sigma2 == 1] = 255
  canny_edges_sigma2 = canny_edges_sigma2.astype(float)
  return gray2rgb(canny_edges_sigma2)

def data_gen(data_dir):
  data_gen = ImageDataGenerator(rotation_range=15, 
                                    width_shift_range=.2, 
                                    height_shift_range=.2, 
                                    brightness_range=[0.8, 1.2],
                                    zoom_range=[0.8, 1.2],
                                    horizontal_flip=True,
                                    preprocessing_function=custom_img_prep)
  data_train = data_gen.flow_from_directory(directory=data_dir + 'train', target_size=(224, 224))
  data_test = data_gen.flow_from_directory(directory=data_dir + 'test', target_size=(224, 224))
  return data_gen, data_train, data_test

def data_gen_samp(data_gen, data_dir):
  samp_img = load_img(data_dir + '/train/Happy/S057_006_00000033.png')
  samp_data_itr = data_gen.flow(np.expand_dims(img_to_array(samp_img), 0), batch_size=1)

  pyplot.figure(figsize=(10,10))
  pyplot.subplot(5, 4, 1)
  pyplot.imshow(samp_img)

  for samp_aug_img in range(19):
    pyplot.subplot(5, 4, samp_aug_img+2)
    pyplot.imshow(samp_data_itr.next()[0].astype('uint8'))
  pyplot.show()


def get_model():
  model = Sequential()

  model.add(Conv2D(input_shape=(224,224,3),filters=64,kernel_size=(5,5),padding="same", activation="relu"))
  model.add(Conv2D(filters=64,kernel_size=(5,5),padding="same", activation="relu"))
  model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

  model.add(Conv2D(filters=128, kernel_size=(5,5), padding="same", activation="relu"))
  model.add(Conv2D(filters=128, kernel_size=(5,5), padding="same", activation="relu"))
  model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

  model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
  model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
  model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

  model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
  model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
  model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

  model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
  model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
  model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
  model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

  model.add(Flatten())
  model.add(Dense(units=4096,activation="relu"))
  model.add(Dense(units=2048,activation="relu"))
  model.add(Dense(units=5, activation="softmax"))

  model.compile(optimizer=Adam(lr=0.001), loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])

  model.summary()

  return model

def train_model(model, data_train, data_test):
  check_point = ModelCheckpoint("emotion_detection_custom_arch1.h5", monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)

  early_stopping = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=10, verbose=1, mode='auto')

  fitting_op = model.fit_generator(steps_per_epoch=50, generator=data_train, validation_data= data_test, validation_steps=10, epochs=100, callbacks=[check_point, early_stopping])

  return fitting_op

data_gen, data_train, data_test = data_gen(data_dir)

data_gen_samp(data_gen, data_dir)

model = get_model()

fitting_op = train_model(model, data_train, data_test)

pyplot.plot(fitting_op.history["accuracy"])
pyplot.plot(fitting_op.history["val_accuracy"])
pyplot.plot(fitting_op.history["loss"])
pyplot.plot(fitting_op.history["val_loss"])
pyplot.title("model accuracy")
pyplot.ylabel("Accuracy")
pyplot.xlabel("Epoch")
pyplot.legend(["Accuracy","Validation Accuracy","loss","Validation Loss"])
pyplot.show()