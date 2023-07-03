import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Set the paths to your dataset folders
train_dir = "DATA"
#test_dir = "path/to/test_data"

# Set the batch size and image size
batch_size = 32
image_size = (400, 400)

# Use ImageDataGenerator for data preprocessing
train_datagen = ImageDataGenerator(rescale=1.0/255)

# Load the training data
train_data = train_datagen.flow_from_directory(train_dir, target_size=image_size, batch_size=batch_size, class_mode='categorical')

# Determine the number of classes (labels)
num_classes = len(train_data.class_indices)

# Define the model architecture
model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(image_size[0], image_size[1], 3)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(128, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_data, epochs=10)

# Save the model as an HDF5 file
model.save("ourmodel.h5")

ind = 0

# Save the labels to a text file
with open("ourlabels.txt", "w") as file:
    for class_name, class_index in train_data.class_indices.items():
        file.write(ind +" "+class_name + "\n")

print("Model and labels saved.")
