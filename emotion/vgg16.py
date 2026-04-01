#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Custom VGG16-like architecture for emotion classification.

@author: abhishek
"""

import os
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2


data_dir = 'data/'

print(os.listdir("data/"))

# Image augmentation techniques using ImageDataGenerator
data_gen = ImageDataGenerator(rescale=1./255,
                              rotation_range=15,
                              width_shift_range=.2,
                              height_shift_range=.2,
                              brightness_range=[0.8, 1.2],
                              zoom_range=[0.8, 1.2],
                              horizontal_flip=True)

samp_img = load_img(data_dir + '/train/Angry/3.jpg')
samp_data_itr = data_gen.flow(np.expand_dims(img_to_array(samp_img), 0), batch_size=1)

plt.figure(figsize=(10, 10))
plt.subplot(5, 4, 1)
plt.imshow(samp_img)

for samp_aug_img in range(19):
    plt.subplot(5, 4, samp_aug_img + 2)
    plt.imshow(samp_data_itr.next()[0].astype('uint8'))
plt.show()

data_train = data_gen.flow_from_directory(directory=data_dir + 'train', target_size=(224, 224))
data_test = data_gen.flow_from_directory(directory=data_dir + 'validation', target_size=(224, 224))

# VGG16-like model architecture
model = Sequential()

model.add(Conv2D(input_shape=(224, 224, 3), filters=64, kernel_size=(5, 5), padding="same", activation="relu"))
model.add(Conv2D(filters=64, kernel_size=(5, 5), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Conv2D(filters=128, kernel_size=(5, 5), padding="same", activation="relu"))
model.add(Conv2D(filters=128, kernel_size=(5, 5), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Flatten())
model.add(Dense(units=4096, activation="relu", kernel_regularizer=l2(1e-4)))
model.add(Dropout(0.5))
model.add(Dense(units=2048, activation="relu", kernel_regularizer=l2(1e-4)))
model.add(Dropout(0.5))
model.add(Dense(units=6, activation="softmax"))

model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

check_point = ModelCheckpoint("emotion_detection_custom_arch1.h5", monitor='val_accuracy',
                              verbose=1, save_best_only=True, save_weights_only=False,
                              mode='auto', save_freq='epoch')

early_stopping = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=10, verbose=1, mode='auto')
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1)

fitting_op = model.fit(data_train, steps_per_epoch=100, validation_data=data_test,
                       validation_steps=10, epochs=100, callbacks=[check_point, early_stopping, lr_scheduler])

plt.plot(fitting_op.history["accuracy"])
plt.plot(fitting_op.history["val_accuracy"])
plt.plot(fitting_op.history["loss"])
plt.plot(fitting_op.history["val_loss"])
plt.title("model accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["Accuracy", "Validation Accuracy", "loss", "Validation Loss"])
plt.show()
