import os
import cv2
import shutil
import numpy as np
from insightface.app import FaceAnalysis
from datetime import datetime
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import logging

# Suppress warnings and logs
logging.getLogger('insightface').setLevel(logging.ERROR)

# Paths to directories
verified_save_path = r"E:\DLC project paper\Webcam\pythonProject\Verified"
model_save_path = r"E:\DLC project paper\Webcam\pythonProject"
os.makedirs(verified_save_path, exist_ok=True)
os.makedirs(model_save_path, exist_ok=True)

# Function to prepare the dataset
def prepare_dataset(dataset_path, target_size=(160, 160), batch_size=32):
    datagen = ImageDataGenerator(rescale=1.0/255, validation_split=0.2)

    train_generator = datagen.flow_from_directory(
        dataset_path,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )

    val_generator = datagen.flow_from_directory(
        dataset_path,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )

    return train_generator, val_generator

# Function to build the model
def build_model(num_classes):
    base_model = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(160, 160, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    output = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=output)
    return model

# Function to calculate dynamic epochs
def calculate_dynamic_epochs(dataset_path, min_epochs=10, factor=1000):
    total_images = sum([len(files) for _, _, files in os.walk(dataset_path)])
    return max(min_epochs, total_images // factor)

# Function to train the model
def train_model(dataset_path, model_save_path):
    train_generator, val_generator = prepare_dataset(dataset_path)
    num_classes = len(train_generator.class_indices)

    model = build_model(num_classes)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Calculate dynamic epochs
    epochs = calculate_dynamic_epochs(dataset_path)
    print(f"Using {epochs} epochs for training.")

    model.fit(train_generator, epochs=epochs, validation_data=val_generator)

    model_file = os.path.join(model_save_path, f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.h5")
    model.save(model_file)
    print(f"Model saved at {model_file}")

# Function to retrain the model with new data
def retrain_model(new_images_path, current_model_path):
    dataset_path = verified_save_path  # Use the verified dataset for retraining

    # Merge new images into the dataset
    for user_folder in os.listdir(new_images_path):
        src_folder = os.path.join(new_images_path, user_folder)
        dest_folder = os.path.join(dataset_path, user_folder)

        os.makedirs(dest_folder, exist_ok=True)
        for file_name in os.listdir(src_folder):
            shutil.copy(os.path.join(src_folder, file_name), dest_folder)

    # Load the existing model or build a new one
    if os.path.exists(current_model_path):
        model = load_model(current_model_path)
        print("Loaded existing model for retraining.")
    else:
        print("No existing model found. Creating a new one.")
        train_generator, _ = prepare_dataset(dataset_path)
        num_classes = len(train_generator.class_indices)
        model = build_model(num_classes)

    # Retrain the model
    train_model(dataset_path, model_save_path)

if __name__ == "__main__":
    # Example usage of retraining

    new_images_path = r"C:\Users\saifu\OneDrive - American International University-Bangladesh\Desktop\Face recognition\Face Data with Model\Verified2"  # Directory containing new images
    current_model_path = os.path.join(model_save_path, "latest_model.h5")

    retrain_model(new_images_path, current_model_path)
