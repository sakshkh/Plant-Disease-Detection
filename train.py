import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import numpy as np

# Clone Dataset
!git clone https://github.com/spMohanty/PlantVillage-Dataset.git
os.makedirs("PlantVillage/train", exist_ok=True)
os.system("mv PlantVillage-Dataset/raw/color/* PlantVillage/train/")
print(" Dataset Cloned and Organized!")

# Data Preprocessing
train_dir = "PlantVillage/train"
batch_size = 8
image_size = (128, 128)

train_datagen = ImageDataGenerator(rescale=1.0/255.0, validation_split=0.2)
train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=image_size, batch_size=batch_size, class_mode="categorical", subset="training"
)
val_generator = train_datagen.flow_from_directory(
    train_dir, target_size=image_size, batch_size=batch_size, class_mode="categorical", subset="validation"
)
print(f" Classes: {train_generator.class_indices.keys()}")

# Build Model
model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(128, 128, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation="relu"),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(256, activation="relu"),
    Dropout(0.5),
    Dense(len(train_generator.class_indices), activation="softmax")
])

# Compile Model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
print(" Model Created Successfully!")

# Training
history = model.fit(train_generator, validation_data=val_generator, epochs=5)

# Save Model
model.save("plant_disease_model.keras")
print(" Model Saved Successfully!")
