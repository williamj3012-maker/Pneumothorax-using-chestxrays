# Advanced Pneumothorax Segmentation

import numpy as np
import cv2
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from model import build_model, advanced_loss_function

# CLAHE preprocessing function

def apply_clahe(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)

# Load dataset
# Make sure to replace this with your actual dataset loading logic

def load_data(data_dir):
    images = []
    masks = []
    for filename in os.listdir(data_dir):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image = cv2.imread(os.path.join(data_dir, filename), cv2.IMREAD_GRAYSCALE)
            mask = cv2.imread(os.path.join(data_dir, 'masks', filename), cv2.IMREAD_GRAYSCALE)
            images.append(apply_clahe(image))
            masks.append(mask)
    return np.array(images), np.array(masks)

# Training settings
DATA_DIR = 'path/to/dataset'
images, masks = load_data(DATA_DIR)
X_train, X_val, y_train, y_val = train_test_split(images, masks, test_size=0.2, random_state=42)

# Build and train the model
model = build_model()
model.compile(optimizer='adam', loss=advanced_loss_function, metrics=['accuracy'])
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=16)

# Save the model
model.save('pneumothorax_segmentation_model.h5')