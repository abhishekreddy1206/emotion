#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VGG16 feature extraction with Random Forest and XGBoost classifiers.
Uses CK+48 dataset with train/validation split.

@author: abhishek
"""

import numpy as np
import matplotlib.pyplot as plt
import glob
import cv2
import os
import pickle

import seaborn as sns

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.utils import to_categorical

from sklearn import preprocessing, metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix


print(os.listdir("CK+48/data/"))

SIZE = 224

# Capture training data and labels
train_images = []
train_labels = []

for directory_path in glob.glob("CK+48/data/train/*"):
    label = os.path.basename(directory_path)
    for img_path in glob.glob(os.path.join(directory_path, "*.*")):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            continue
        img = cv2.resize(img, (SIZE, SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        train_images.append(img)
        train_labels.append(label)

train_images = np.array(train_images)
train_labels = np.array(train_labels)


# Capture test/validation data and labels
test_images = []
test_labels = []
for directory_path in glob.glob("CK+48/data/validation/*"):
    label = os.path.basename(directory_path)
    for img_path in glob.glob(os.path.join(directory_path, "*.*")):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            continue
        img = cv2.resize(img, (SIZE, SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        test_images.append(img)
        test_labels.append(label)

test_images = np.array(test_images)
test_labels = np.array(test_labels)

# Encode labels from text to integers.
le = preprocessing.LabelEncoder()
all_labels = np.concatenate([train_labels, test_labels])
le.fit(all_labels)
train_labels_encoded = le.transform(train_labels)
test_labels_encoded = le.transform(test_labels)

# Assign to meaningful convention
x_train, y_train = train_images, train_labels_encoded
x_test, y_test = test_images, test_labels_encoded

# Normalize pixel values to between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0

# One hot encode y values for neural network.
y_train_one_hot = to_categorical(y_train)
y_test_one_hot = to_categorical(y_test)

# Load VGG16 for feature extraction only (all layers frozen)
VGG_model = VGG16(weights='imagenet', include_top=False, input_shape=(SIZE, SIZE, 3))
VGG_model.summary()

for layer in VGG_model.layers:
    layer.trainable = False

VGG_model.summary()

# Extract features using VGG16
feature_extractor = VGG_model.predict(x_train)
features = feature_extractor.reshape(feature_extractor.shape[0], -1)
X_for_training = features

# XGBOOST
import xgboost as xgb
xg_model = xgb.XGBClassifier()
xg_model.fit(X_for_training, y_train)

# Extract test features
X_test_feature = VGG_model.predict(x_test)
X_test_features = X_test_feature.reshape(X_test_feature.shape[0], -1)

# Predict using XGBoost
prediction = xg_model.predict(X_test_features)
prediction = le.inverse_transform(prediction)

# RANDOM FOREST
rf_model = RandomForestClassifier(n_estimators=50, random_state=42)
rf_model.fit(X_for_training, y_train)

# Predict using RF
prediction_RF = rf_model.predict(X_test_features)
prediction_RF = le.inverse_transform(prediction_RF)

# Save models
xg_filename = 'model_XG.sav'
pickle.dump(xg_model, open(xg_filename, 'wb'))
rf_filename = 'RF_model.sav'
pickle.dump(rf_model, open(rf_filename, 'wb'))

# Print overall accuracy
print("XGBoost Accuracy = ", metrics.accuracy_score(test_labels, prediction))
print("RF Accuracy = ", metrics.accuracy_score(test_labels, prediction_RF))

# Confusion Matrix
cm = confusion_matrix(test_labels, prediction)
cm2 = confusion_matrix(test_labels, prediction_RF)

plt.figure()
sns.heatmap(cm, annot=True)
plt.title("XGBoost Confusion Matrix")
plt.show()

plt.figure()
sns.heatmap(cm2, annot=True)
plt.title("Random Forest Confusion Matrix")
plt.show()

# Check results on a few select images
n = np.random.randint(0, x_test.shape[0])
img = x_test[n]
plt.imshow(img)
input_img = np.expand_dims(img, axis=0)
input_img_feature = VGG_model.predict(input_img)
input_img_features = input_img_feature.reshape(input_img_feature.shape[0], -1)
prediction_sample = xg_model.predict(input_img_features)[0]
prediction_sample = le.inverse_transform([prediction_sample])
print("The prediction for this image is: ", prediction_sample)
print("The actual label for this image is: ", test_labels[n])
