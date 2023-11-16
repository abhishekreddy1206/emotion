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

import scikitplot
import seaborn as sns
from matplotlib import pyplot as plt
import skimage.io, skimage, skimage.feature

from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from skimage.color import rgb2gray, gray2rgb

import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Flatten, Dense, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.layers import Dropout, BatchNormalization, Activation
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import plot_model
from keras.applications import VGG19, ResNet50

from keras import backend as K
from keras.utils import np_utils

np.random.seed(42)
DATA = "CK+48"
base_path = "CK+48/images/"
categories = ["Happy", "Fear", "Sadness", "Surprise", "Anger"]
INPUT_SHAPE = (48,48,1)
NUM_CLASSES = len(categories)
class_count = {}
img_width, img_height = 224, 224



fnames = []
for category in categories:
    flower_folder = os.path.join(base_path, category)
    file_names = os.listdir(flower_folder)
    full_path = [os.path.join(flower_folder, file_name) for file_name in file_names]
    fnames.append(full_path)

print('length for each category:', [len(f) for f in fnames])

images = []
for names in fnames:
    one_category_images = [cv2.imread(name) for name in names if (cv2.imread(name)) is not None]
    images.append(one_category_images)
print('number of images for each category:', [len(f) for f in images])

# for i,imgs in enumerate(images):
#     shapes = [img.shape for img in imgs]
#     widths = [shape[0] for shape in shapes]
#     heights = [shape[1] for shape in shapes]
#     print('%d,%d is the min shape for %s' % (np.min(widths), np.min(heights), categories[i]))

plt.figure(figsize=(15,10))
for i, imgs in enumerate(images):
    plt.subplot(2,3,i+1)
    idx = np.random.randint(len(imgs))
    plt.imshow(cv2.cvtColor(imgs[idx].copy(), cv2.COLOR_BGR2RGB))
    plt.grid('off')
    plt.title(categories[i]+' '+str(idx))
plt.show()

img = images[3][5]
print(img.shape)
resized_img = resize(img, (img_width, img_height, 3))
resized_img2 = cv2.resize(img,(img_width, img_height), interpolation = cv2.INTER_CUBIC)
print(resized_img.shape)
plt.figure(figsize=(15,15))
plt.subplot(2,2,1)
plt.title('original image (BGR-channel)')
plt.grid(False)
plt.imshow(img)
plt.subplot(2,2,2)
plt.title('original image (RGB-channel)')
plt.grid(False)
plt.imshow(cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB))
plt.subplot(2,2,3)
plt.title('resized by skimage (BGR-channel)')
plt.grid(False)
plt.imshow((resized_img))
plt.subplot(2,2,4)
plt.title('resized by opencv (RGB-channel)')
plt.grid(False)
plt.imshow(cv2.cvtColor(resized_img2.copy(), cv2.COLOR_BGR2RGB))
plt.show()


resized_images = []
for i,imgs in enumerate(images):
    resized_images.append([cv2.resize(img, (img_width, img_height), interpolation = cv2.INTER_CUBIC) for img in imgs])

train_images = []
val_images = []
for imgs in resized_images:
    train, test = train_test_split(imgs, train_size=0.7, test_size=0.3)
    train_images.append(train)
    val_images.append(test)


len_train_images = [len(imgs) for imgs in train_images]
print(len_train_images)
print('sum of train images:', np.sum(len_train_images))
train_categories = np.zeros((np.sum(len_train_images)), dtype='uint8')
for i in range(5):
    if i == 0:
        train_categories[:len_train_images[i]] = i
    else:
        train_categories[np.sum(len_train_images[:i]):np.sum(len_train_images[:i+1])] = i
        
len_val_images = [len(imgs) for imgs in val_images]
print(len_val_images)
print('sum of val_images:', np.sum(len_val_images))
val_categories = np.zeros((np.sum(len_val_images)), dtype='uint8')
for i in range(5):
    if i == 0:
        val_categories[:len_val_images[i]] = i
    else:
        val_categories[np.sum(len_val_images[:i]):np.sum(len_val_images[:i+1])] = i



tmp_train_imgs = []
tmp_val_imgs = []
for imgs in train_images:
    tmp_train_imgs += imgs
for imgs in val_images:
    tmp_val_imgs += imgs
train_images = np.array(tmp_train_imgs)
val_images = np.array(tmp_val_imgs)



print('train data:', train_images.shape)
print('train labels:', train_categories.shape)

train_data = train_images.astype('float32')
val_data = val_images.astype('float32')
train_labels = np_utils.to_categorical(train_categories, len(categories))
val_labels = np_utils.to_categorical(val_categories, len(categories))
print()
print('After converting')
print('train data:', train_data.shape)
print('train labels:', train_labels.shape)


train_data = train_data[:3400]
train_labels = train_labels[:3400]
val_data = val_data[:860]
val_labels = val_labels[:860]
print('shape of train data:', train_data.shape)
print('shape of train labels:', train_labels.shape)
print('shape of val data:', val_data.shape)
print('shape of val labels:', val_labels.shape)

WEIGHTS_PATH_NO_TOP = 'resnet50_notop.h5'



def custom_img_prep_samp(img):
  img_2d = rgb2gray(img)
  canny_edges = skimage.feature.canny(img_2d)
  canny_edges_sigma2 = skimage.feature.canny(img_2d, sigma=2)
  figure, (img1, img2, img3, img4) = plt.subplots(nrows=1, ncols=4, figsize=(8, 3), sharex=True, sharey=True)

  print(img.dtype)
  print(img_2d.dtype)
  print((gray2rgb(canny_edges)).dtype)
  img1.imshow(img, cmap=plt.cm.gray)
  img2.imshow(canny_edges, cmap=plt.cm.gray)
  img3.imshow(canny_edges_sigma2, cmap=plt.cm.gray)
  canny_edges_sigma2 = gray2rgb(img_2d)
  img4.imshow(canny_edges_sigma2, cmap=plt.cm.gray)

  figure.tight_layout()

  plt.show()

def custom_img_prep(img):
  img_2d = rgb2gray(img)
  canny_edges_sigma2 = skimage.feature.canny(img_2d, sigma=2)
  canny_edges_sigma2 = canny_edges_sigma2.astype(int)
  canny_edges_sigma2[canny_edges_sigma2 == 1] = 255
  canny_edges_sigma2 = canny_edges_sigma2.astype(float)
  return gray2rgb(canny_edges_sigma2)

def return_name(label_arr):
  idx = np.where(label_arr == 1)
  return idx[0][0]

def plot_model_history(model_name, history, epochs):
  plt.figure(figsize=(15, 5))
  
  # summarize history for accuracy
  plt.subplot(1, 2 ,1)
  plt.plot(np.arange(0, len(history['acc'])), history['acc'], 'r')
  plt.plot(np.arange(1, len(history['val_acc'])+1), history['val_acc'], 'g')
  plt.xticks(np.arange(0, epochs+1, epochs/10))
  plt.title('Training Accuracy vs. Validation Accuracy')
  plt.xlabel('Num of Epochs')
  plt.ylabel('Accuracy')
  plt.legend(['train', 'validation'], loc='best')
  
  plt.subplot(1, 2, 2)
  plt.plot(np.arange(1, len(history['loss'])+1), history['loss'], 'r')
  plt.plot(np.arange(1, len(history['val_loss'])+1), history['val_loss'], 'g')
  plt.xticks(np.arange(0, epochs+1, epochs/10))
  plt.title('Training Loss vs. Validation Loss')
  plt.xlabel('Num of Epochs')
  plt.ylabel('Loss')
  plt.legend(['train', 'validation'], loc='best')
  
  
  plt.show()

def predict_one_image(img, model):
  img = cv2.resize(img, (img_width, img_height), interpolation = cv2.INTER_CUBIC)
  img = np.reshape(img, (1, img_width, img_height, 3))
  img = img/255.
  pred = model.predict(img)
  class_num = np.argmax(pred, axis=1)
  return class_num, np.max(pred)

def create_model_from_scratch():
    """
     train from scratch
    """
    model = Sequential()
    model.add(Conv2D(32, (3,3), padding='same', input_shape=train_data.shape[1:], activation='relu', name='conv_1'))
    model.add(Conv2D(32, (3,3), activation='relu', name='conv_2'))
    model.add(MaxPooling2D(pool_size=(2,2), name='maxpool_1'))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3,3), padding='same', activation='relu', name='conv_3'))
    model.add(Conv2D(64, (3,3), activation='relu', name='conv_4'))
    model.add(MaxPooling2D(pool_size=(2,2), name='maxpool_2'))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(128, (3,3), padding='same', activation='relu', name='conv_5'))
    model.add(Conv2D(128, (3,3), activation='relu', name='conv_6'))
    model.add(MaxPooling2D(pool_size=(2,2), name='maxpool_3'))

    model.add(Flatten())
    model.add(Dense(512, activation='relu', name='dense_1'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu', name='dense_2'))
    model.add(Dense(len(categories), name='output'))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc']) # optimizer=RMSprop(lr=0.001)
    
    return model

def build_cnn(input_shape, num_classes, show_summary=True):
    model_in = Input(shape=input_shape, name="input_CNN")
    
    conv2d_1 = Conv2D(
        filters=64,
        kernel_size=(3,3),
        activation='elu',
        padding='same',
        kernel_initializer='he_normal',
        name='conv2d_1'
    )(model_in)
    batchnorm_1 = BatchNormalization(name='batchnorm_1')(conv2d_1)
    conv2d_2 = Conv2D(
        filters=64,
        kernel_size=(3,3),
        activation='elu',
        padding='same',
        kernel_initializer='he_normal',
        name='conv2d_2'
    )(batchnorm_1)
    batchnorm_2 = BatchNormalization(name='batchnorm_2')(conv2d_2)
    
    maxpool2d_1 = MaxPooling2D(pool_size=(2,2), name='maxpool2d_1')(batchnorm_2)
    dropout_1 = Dropout(0.35, name='dropout_1')(maxpool2d_1)

    conv2d_3 = Conv2D(
        filters=128,
        kernel_size=(3,3),
        activation='elu',
        padding='same',
        kernel_initializer='he_normal',
        name='conv2d_3'
    )(dropout_1)
    batchnorm_3 = BatchNormalization(name='batchnorm_3')(conv2d_3)
    conv2d_4 = Conv2D(
        filters=128,
        kernel_size=(3,3),
        activation='elu',
        padding='same',
        kernel_initializer='he_normal',
        name='conv2d_4'
    )(batchnorm_3)
    batchnorm_4 = BatchNormalization(name='batchnorm_4')(conv2d_4)
    
    maxpool2d_2 = MaxPooling2D(pool_size=(2,2), name='maxpool2d_2')(batchnorm_4)
    dropout_2 = Dropout(0.4, name='dropout_2')(maxpool2d_2)

    conv2d_5 = Conv2D(
        filters=256,
        kernel_size=(3,3),
        activation='elu',
        padding='same',
        kernel_initializer='he_normal',
        name='conv2d_5'
    )(dropout_2)
    batchnorm_5 = BatchNormalization(name='batchnorm_5')(conv2d_5)
    conv2d_6 = Conv2D(
        filters=256,
        kernel_size=(3,3),
        activation='elu',
        padding='same',
        kernel_initializer='he_normal',
        name='conv2d_6'
    )(batchnorm_5)
    batchnorm_6 = BatchNormalization(name='batchnorm_6')(conv2d_6)
    
    maxpool2d_3 = MaxPooling2D(pool_size=(2,2), name='maxpool2d_3')(batchnorm_6)
    dropout_3 = Dropout(0.5, name='dropout_3')(maxpool2d_3)

    flatten = Flatten(name='flatten')(dropout_3)
        
    dense_1 = Dense(
        256,
        activation='elu',
        kernel_initializer='he_normal',
        name='dense1'
    )(flatten)
    batchnorm_7 = BatchNormalization(name='batchnorm_7')(dense_1)
    dropout_4 = Dropout(0.6, name='dropout_4')(batchnorm_7)

    model_out = Dense(
        num_classes,
        activation="softmax",
        name="ouput_CNN"
    )(dropout_4)

    model = Model(inputs=model_in, outputs=model_out)
    
    if show_summary:
        model.summary()
    
    return model

def create_model_from_VGG19():
    #Load model wothout classifier/fully connected layers
    model = VGG19(weights = "imagenet", include_top=False, input_shape = (img_width, img_height, 3))
    
    for layer in model.layers[:1]:
      layer.trainable = False
    x = model.output
    x = Flatten()(x)
    x = Dense(1024, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation="relu")(x)
    predictions = Dense(len(categories), activation="softmax")(x)
    
    final_model = Model(inputs = model.input, outputs = predictions)
    final_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    
    return final_model

def create_model_from_ResNet50():
    #Load model wothout classifier/fully connected layers
    model = Sequential()
    model.add(ResNet50(include_top=False, pooling='avg', weights='imagenet'))
    model.add(Flatten())
    model.add(BatchNormalization())
    model.add(Dense(2048, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(1024, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(len(categories), activation='softmax'))
    model.layers[0].trainable = False
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc']) # optimizer=RMSprop(lr=0.001)
    
    return model


model_VGG19 = create_model_from_VGG19()
model_VGG19.summary()
model_ResNet50 = create_model_from_ResNet50()
model_ResNet50.summary()


# Parameters
batch_size = 32
epochs1 = 50
epochs2 = 50
epochs3 = 50


train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.3,
    horizontal_flip=False
)

train_datagen2 = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.3,
    horizontal_flip=False,
    preprocessing_function=custom_img_prep
)

val_datagen = ImageDataGenerator(
    rescale=1./255,
)

train_generator = train_datagen.flow(
    train_data,
    train_labels,
    batch_size=batch_size
)

train_generator2 = train_datagen2.flow(
    train_data,
    train_labels,
    batch_size=batch_size
)

val_generator = val_datagen.flow(
    val_data,
    val_labels,
    batch_size=batch_size
)


start = time.time()

#XGBOOST
import xgboost as xgb
xgb_model = xgb.XGBClassifier()


model_VGG19_info = model_VGG19.fit_generator(
    generator=train_generator, 
    steps_per_epoch=len(train_data)/batch_size,   
    epochs=50, 
    validation_steps=len(val_data)/batch_size, 
    validation_data=val_generator,
    verbose=2
)


end = time.time()
duration = end - start
print ('\n model_VGG19 took %0.2f seconds (%0.1f minutes) to train for %d epochs'%(duration, duration/60, epochs2) )

model_VGG19_info_canny = model_VGG19.fit_generator(
    generator=train_generator2, 
    steps_per_epoch=len(train_data)/batch_size,
    epochs=50, 
    validation_steps=len(val_data)/batch_size, 
    validation_data=val_generator,
    verbose=2
)

model_ResNet50_info = model_ResNet50.fit_generator(
    generator=train_generator, 
    steps_per_epoch=len(train_data)/batch_size,  
    epochs=50, 
    validation_steps=len(val_data)/batch_size, 
    validation_data=val_generator, 
    verbose=2
)


end = time.time()
duration = end - start
print ('\n model_ResNet50 took %0.2f seconds (%0.1f minutes) to train for %d epochs'%(duration, duration/60, epochs3) )

model_ResNet50_canny_info = model_ResNet50.fit_generator(
    generator=train_generator2, 
    steps_per_epoch=len(train_data)/batch_size,  
    epochs=50, 
    validation_steps=len(val_data)/batch_size, 
    validation_data=val_generator, 
    verbose=2
)


# plot_model_history('model_scratch', model_scratch_info.history, epochs1)
plot_model_history('model_VGG19', model_VGG19_info.history, epochs2)
plot_model_history('model_ResNet50', model_ResNet50_info.history, epochs3)


#extract features VGG19
feature_extractor_vgg=model_VGG19_info_canny.predict(train_data)
features = feature_extractor_vgg.reshape(feature_extractor_vgg.shape[0], -1)
X_vgg_for_training = features
#extract features Resnet
feature_extractor_resnet=model_ResNet50_canny_info.predict(train_data)
features = feature_extractor_resnet.reshape(feature_extractor_resnet.shape[0], -1)
X_resnet_for_training = features

#apply xgboost on vgg and resnet
xgb_model.fit(X_vgg_for_training, val_data) 
X_vgg_test_feature = model_VGG19_info_canny.predict(val_data)
X_vgg_test_features = X_vgg_test_feature.reshape(X_vgg_test_feature.shape[0], -1)
xgb_model.fit(X_resnet_for_training, val_data) 
X_resnet_test_feature = model_ResNet50_canny_info.predict(val_data)
X_resnet_test_features = X_resnet_test_feature.reshape(X_resnet_test_feature.shape[0], -1)


test_dataset = tf.data.Dataset.from_tensor_slices((train_data, val_data))
test_dataset = test_dataset.batch(64)
model_ResNet50.evaluate(val_data)
test_img = cv2.imread('CK+48/images/Happy/S010_006_00000014.png')
pred, probability = predict_one_image(test_img, model_ResNet50)
pred_2= np.argmax(pred)
print('%s %d%%' % (categories[pred[0]], round(probability, 2) * 100))
print(classification_report(pred_2, categories))
_, ax = plt.subplots(1)
plt.imshow(cv2.cvtColor(test_img.copy(), cv2.COLOR_BGR2RGB))
# Turn off tick labels
ax.set_yticklabels([])
ax.set_xticklabels([])
plt.grid('off')
plt.show()


test_img = cv2.imread('CK+48/images/Sadness/54.jpg')
pred, probability = predict_one_image(test_img, model_ResNet50)
print('%s %d%%' % (categories[pred], round(probability, 2) * 100))
_, ax = plt.subplots(1)
plt.imshow(cv2.cvtColor(test_img.copy(), cv2.COLOR_BGR2RGB))
# Turn off tick labels
ax.set_yticklabels([])
ax.set_xticklabels([])
plt.grid('off')
plt.show()


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
# print(f"\ntotal images are {total_images}")

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

text_to_label = dict((v,k) for k,v in label_to_text.items())

num_emotions = len(categories)
fig = plt.figure(1, (num_emotions*1.5, num_emotions*1.5))

idx = 0
for k in label_to_text:
    sample_indices = np.random.choice(np.where(img_label[:,k]==1)[0], size=4, replace=False)
    sample_images = img_arr[sample_indices]
    for img in sample_images:
        idx += 1
        ax = plt.subplot(num_emotions,4,idx)
        ax.imshow(img.reshape(48,48), cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(label_to_text[k])
        plt.tight_layout()


img_arr = img_arr / 255.
X_train, X_test, y_train, y_test = train_test_split(img_arr, img_label,
                                                    shuffle=True, stratify=img_label,
                                                    train_size=0.7, random_state=42)
X_train.shape, X_test.shape, y_train.shape, y_test.shape


model = build_cnn(INPUT_SHAPE, NUM_CLASSES, show_summary=False)
plot_model(model, show_shapes=True, show_layer_names=True, expand_nested=True, dpi=50,)


train_datagen_1 = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
)

train_datagen_2 = ImageDataGenerator(
    rotation_range=8,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.08,
    horizontal_flip=True,
    preprocessing_function=custom_img_prep
)

early_stopping_1 = EarlyStopping(
    monitor='val_accuracy',
    min_delta=0.00005,
    patience=11,
    verbose=1,
    restore_best_weights=True,
)

early_stopping_2 = EarlyStopping(
    monitor='accuracy',
    min_delta=0.0005,
    patience=15,
    verbose=1,
    restore_best_weights=True,
)

lr_scheduler_1 = ReduceLROnPlateau(
    monitor='val_accuracy',
    min_delta=0.0001,
    factor=0.4,
    patience=5,
    min_lr=1e-7,
    verbose=1,
)

lr_scheduler_2 = ReduceLROnPlateau(
    monitor='val_accuracy',
    min_delta=0.0001,
    factor=0.5,
    patience=4,
    min_lr=1e-7,
    verbose=1,
)

BEST_CONFIG = {
        "batch_size": 12,
        "epochs": 100,
        "loss": "categorical_crossentropy",
        "optim": optimizers.Adam(0.01),
        "train_datagen": train_datagen_2,
        "lr_scheduler": lr_scheduler_2,
        "callbacks": [lr_scheduler_2, early_stopping_2],
    }

model.compile(
        loss=BEST_CONFIG["loss"],
        optimizer=BEST_CONFIG["optim"],
        metrics=['accuracy']
)


BEST_CONFIG["train_datagen"].fit(X_train)
history = model.fit(
    BEST_CONFIG["train_datagen"].flow(X_train, y_train, batch_size=BEST_CONFIG["batch_size"]),
    validation_data=(X_test, y_test),
    steps_per_epoch=len(X_train) / BEST_CONFIG["batch_size"],
    epochs=BEST_CONFIG["epochs"],
    callbacks=BEST_CONFIG["callbacks"],
)


#extract features custom CNN
feature_extractor_cnn=model.predict(X_train)
features = feature_extractor_cnn.reshape(feature_extractor_cnn.shape[0], -1)
X_cnn_for_training = features
xgb_model.fit(X_cnn_for_training, y_train) 
X_cnn_test_feature = model.predict(val_data)
X_cnn_test_features = X_cnn_test_feature.reshape(X_cnn_test_feature.shape[0], -1)

sns.set()
fig = plt.figure(0, (12, 4))

ax = plt.subplot(1, 2, 1)
sns.lineplot(history.epoch, history.history['accuracy'], label='train')
sns.lineplot(history.epoch, history.history['val_accuracy'], label='valid')
plt.title('Accuracy')
plt.tight_layout()

ax = plt.subplot(1, 2, 2)
sns.lineplot(history.epoch, history.history['loss'], label='train')
sns.lineplot(history.epoch, history.history['val_loss'], label='valid')
plt.title('Loss')
plt.tight_layout()
plt.show()


yhat_test = np.argmax(model.predict(X_test), axis=1)
ytest_ = np.argmax(y_test, axis=1)

scikitplot.metrics.plot_confusion_matrix(ytest_, yhat_test, figsize=(7,7))

test_accu = np.sum(ytest_ == yhat_test) / len(ytest_) * 100
print(f"test accuracy: {round(test_accu, 4)} %\n\n")

print(classification_report(ytest_, yhat_test))


yhat_train = np.argmax(model.predict(X_train), axis=1)
ytrain_ = np.argmax(y_train, axis=1)

train_accu = np.sum(ytrain_ == yhat_train) / len(ytrain_) * 100
print(f"train accuracy: {round(train_accu, 4)} %")


f = 1
for emotion in categories:
    emotion_imgs = np.random.choice(np.where(y_test[:, text_to_label[emotion]]==1)[0], size=8, replace=False)

    plt.figure(f, (16,1.5))
    f += 1

    for i,img_idx in enumerate(emotion_imgs):
        sample_img = X_test[img_idx,:,:,0]
        pred = label_to_text[np.argmax(model.predict(sample_img.reshape(1,48,48,1)), axis=1)[0]]
        ax = plt.subplot(1, 9, i+1)
        ax.imshow(sample_img, cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"t:{emotion[:3]}, p:{pred[:3]}")
        plt.tight_layout()
