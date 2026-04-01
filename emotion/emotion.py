#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VGG16 feature extraction with Random Forest and XGBoost classifiers.
Uses aligned face dataset.

@author: abhishek
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import cv2
import pickle
import os

import seaborn as sns

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.utils import to_categorical

from sklearn import preprocessing, metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score


# Read input images and assign labels based on folder names
print(os.listdir("aligned/"))

SIZE = 224  # Resize images

# Capture training data and labels into respective lists
train_images = []
train_labels = []

for directory_path in glob.glob("aligned/train/*"):
    label = os.path.basename(directory_path)
    for img_path in glob.glob(os.path.join(directory_path, "*.jpg")):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (SIZE, SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        train_images.append(img)
        train_labels.append(label)

train_images = np.array(train_images)
train_labels = np.array(train_labels)


# Capture test/validation data and labels into respective lists
test_images = []
test_labels = []
for directory_path in glob.glob("aligned/test/*"):
    label = os.path.basename(directory_path)
    for img_path in glob.glob(os.path.join(directory_path, "*.jpg")):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
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
x_train, y_train, x_test, y_test = train_images, train_labels_encoded, test_images, test_labels_encoded

# Normalize pixel values to between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0

# One hot encode y values for neural network.
y_train_one_hot = to_categorical(y_train)
y_test_one_hot = to_categorical(y_test)

# Load VGG16 for feature extraction with GlobalAveragePooling2D (512 dims vs 25,088)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(SIZE, SIZE, 3))
for layer in base_model.layers:
    layer.trainable = False
x = GlobalAveragePooling2D()(base_model.output)
VGG_model = Model(inputs=base_model.input, outputs=x)
VGG_model.summary()

# Extract features using VGG16 (now 512-dim vectors)
X_for_training = VGG_model.predict(x_train)

# XGBOOST
import xgboost as xgb
xg_model = xgb.XGBClassifier()
xg_model.fit(X_for_training, y_train)

# Send test data through same feature extractor process (already 512-dim)
X_test_features = VGG_model.predict(x_test)

# Now predict using the trained XGBoost model.
prediction = xg_model.predict(X_test_features)
prediction = le.inverse_transform(prediction)


# RANDOM FOREST
model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X_for_training, y_train)

# Now predict using the trained RF model.
prediction_RF = model.predict(X_test_features)
prediction_RF = le.inverse_transform(prediction_RF)


# Save models for future use
xg_filename = 'model_XG.sav'
pickle.dump(xg_model, open(xg_filename, 'wb'))
rf_filename = 'RF_model.sav'
pickle.dump(model, open(rf_filename, 'wb'))


# Print overall accuracy
print("XGBoost Accuracy = ", metrics.accuracy_score(test_labels, prediction))
print("RF Accuracy = ", metrics.accuracy_score(test_labels, prediction_RF))

# Cross-validation scores
import xgboost as xgb
all_features = np.vstack([X_for_training, X_test_features])
all_labels_encoded = np.concatenate([y_train, y_test])
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
xgb_cv_scores = cross_val_score(xgb.XGBClassifier(), all_features, all_labels_encoded, cv=cv, scoring='accuracy')
rf_cv_scores = cross_val_score(RandomForestClassifier(n_estimators=50, random_state=42), all_features, all_labels_encoded, cv=cv, scoring='accuracy')
print(f"XGBoost 5-fold CV: {xgb_cv_scores.mean():.4f} +/- {xgb_cv_scores.std():.4f}")
print(f"RF 5-fold CV: {rf_cv_scores.mean():.4f} +/- {rf_cv_scores.std():.4f}")

# Confusion Matrix - verify accuracy of each class
cm = confusion_matrix(test_labels, prediction)
cm2 = confusion_matrix(test_labels, prediction_RF)
sns.heatmap(cm, annot=True)
plt.title("XGBoost Confusion Matrix")
plt.show()

sns.heatmap(cm2, annot=True)
plt.title("Random Forest Confusion Matrix")
plt.show()

# Check results on a few select images
n = np.random.randint(0, x_test.shape[0])
img = x_test[n]
plt.imshow(img)
input_img = np.expand_dims(img, axis=0)
input_img_features = VGG_model.predict(input_img)
prediction = xg_model.predict(input_img_features)[0]
prediction = le.inverse_transform([prediction])
print("The prediction for this image is: ", prediction)
print("The actual label for this image is: ", test_labels[n])


# Load saved model for verification
loaded_model = pickle.load(open(xg_filename, 'rb'))
