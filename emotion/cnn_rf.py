#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CNN feature extraction with Random Forest classifier.
Uses CK+48 dataset.

@author: abhishek
"""

import numpy as np
import matplotlib.pyplot as plt
import glob
import cv2
import os

import seaborn as sns

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2

from sklearn import preprocessing, metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score


print(os.listdir("CK+48/data/"))

SIZE = 128

train_images = []
train_labels = []
for directory_path in glob.glob("CK+48/data/train/*"):
    label = os.path.basename(directory_path)
    print(label)
    for img_path in glob.glob(os.path.join(directory_path, "*.*")):
        print(img_path)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (SIZE, SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        train_images.append(img)
        train_labels.append(label)

train_images = np.array(train_images)
train_labels = np.array(train_labels)


# test
test_images = []
test_labels = []
for directory_path in glob.glob("CK+48/data/validation/*"):
    label = os.path.basename(directory_path)
    for img_path in glob.glob(os.path.join(directory_path, "*.*")):
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

# Split data into test and train datasets
x_train, y_train, x_test, y_test = train_images, train_labels_encoded, test_images, test_labels_encoded

# Normalize pixel values to between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0

# One hot encode y values for neural network.
y_train_one_hot = to_categorical(y_train)
y_test_one_hot = to_categorical(y_test)

num_classes = len(le.classes_)

activation = 'elu'

feature_extractor = Sequential()
feature_extractor.add(Conv2D(32, 3, activation=activation, padding='same', input_shape=(SIZE, SIZE, 3)))
feature_extractor.add(BatchNormalization())

feature_extractor.add(Conv2D(32, 3, activation=activation, padding='same', kernel_initializer='he_uniform'))
feature_extractor.add(BatchNormalization())
feature_extractor.add(MaxPooling2D())

feature_extractor.add(Conv2D(64, 3, activation=activation, padding='same', kernel_initializer='he_uniform'))
feature_extractor.add(BatchNormalization())

feature_extractor.add(Conv2D(64, 3, activation=activation, padding='same', kernel_initializer='he_uniform'))
feature_extractor.add(BatchNormalization())
feature_extractor.add(MaxPooling2D())

feature_extractor.add(Conv2D(128, 3, activation=activation, padding='same', kernel_initializer='he_uniform'))
feature_extractor.add(BatchNormalization())

feature_extractor.add(Conv2D(128, 3, activation=activation, padding='same', kernel_initializer='he_uniform'))
feature_extractor.add(BatchNormalization())
feature_extractor.add(MaxPooling2D())

feature_extractor.add(Flatten())

# Add layers for deep learning prediction
x = feature_extractor.output
x = Dense(128, activation=activation, kernel_initializer='he_uniform', kernel_regularizer=l2(1e-4))(x)
prediction_layer = Dense(num_classes, activation='softmax')(x)

# Make a new model combining both feature extractor and x
cnn_model = Model(inputs=feature_extractor.input, outputs=prediction_layer)
cnn_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
print(cnn_model.summary())

# Train the CNN model
cnn_callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1),
]
history = cnn_model.fit(x_train, y_train_one_hot, epochs=50, validation_data=(x_test, y_test_one_hot),
                        callbacks=cnn_callbacks)


# plot the training and validation accuracy and loss at each epoch
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
plt.plot(epochs, acc, 'y', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


prediction_NN = cnn_model.predict(x_test)
prediction_NN = np.argmax(prediction_NN, axis=-1)
prediction_NN = le.inverse_transform(prediction_NN)

# Confusion Matrix - verify accuracy of each class
cm = confusion_matrix(test_labels, prediction_NN)
print(cm)
sns.heatmap(cm, annot=True)

# Check results on a few select images
n = 9
img = x_test[n]
plt.imshow(img)
input_img = np.expand_dims(img, axis=0)
prediction = np.argmax(cnn_model.predict(input_img))
prediction = le.inverse_transform([prediction])
print("The prediction for this image is: ", prediction)
print("The actual label for this image is: ", test_labels[n])

# Now, let us use features from convolutional network for RF
X_for_RF = feature_extractor.predict(x_train)

# RANDOM FOREST
RF_model = RandomForestClassifier(n_estimators=50, random_state=42)
RF_model.fit(X_for_RF, y_train)

# Send test data through same feature extractor process
X_test_feature = feature_extractor.predict(x_test)
# Now predict using the trained RF model.
prediction_RF = RF_model.predict(X_test_feature)
prediction_RF = le.inverse_transform(prediction_RF)

# Print overall accuracy
print("Accuracy = ", metrics.accuracy_score(test_labels, prediction_RF))

# Cross-validation scores
all_features_cv = np.vstack([X_for_RF, X_test_feature])
all_labels_cv = np.concatenate([y_train, y_test])
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
rf_cv_scores = cross_val_score(RandomForestClassifier(n_estimators=50, random_state=42), all_features_cv, all_labels_cv, cv=cv, scoring='accuracy')
print(f"RF 5-fold CV: {rf_cv_scores.mean():.4f} +/- {rf_cv_scores.std():.4f}")

# Confusion Matrix - verify accuracy of each class
cm = confusion_matrix(test_labels, prediction_RF)
sns.heatmap(cm, annot=True)

# Check results on a few select images
n = 9
img = x_test[n]
plt.imshow(img)
input_img = np.expand_dims(img, axis=0)
input_img_features = feature_extractor.predict(input_img)
prediction_RF = RF_model.predict(input_img_features)[0]
prediction_RF = le.inverse_transform([prediction_RF])
print("The prediction for this image is: ", prediction_RF)
print("The actual label for this image is: ", test_labels[n])
