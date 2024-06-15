import os
import cv2
import numpy as np
import tensorflow as tf
from keras import layers, models
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import matplotlib.pyplot as plt


#define tham so
epochs = 50
folder_train_path = r'C:/Users/Windows/Desktop/Nhan dien bien so xe/dataset/Ky tu bien so'
number_class = 30 # 10 digits + 20 letters


# Function to load and preprocess the data
def load_and_preprocess_data(folder_path):
    # Initialize lists to store images and labels
    images = []
    labels = []
    
    # Loop through each folder in the dataset directory
    for folder_name in os.listdir(folder_path):
        folder = os.path.join(folder_path, folder_name)
        if os.path.isdir(folder):
            # Get the label (class) from the folder name
            label = int(folder_name)
            # Loop through each image in the folder
            for filename in os.listdir(folder):
                img_path = os.path.join(folder, filename)
                # Read the image and resize it to (227, 227) as required by AlexNet
                img = cv2.imread(img_path)
                img = cv2.resize(img, (227, 227))
                # Normalize the pixel values to be between 0 and 1
                img = img / 255.0
                # Append the image and label to the lists
                images.append(img)
                labels.append(label)
                
    # Convert the lists to numpy arrays
    images = np.array(images)
    labels = np.array(labels)
    
    # Convert labels to one-hot encoded vectors
    labels = to_categorical(labels, num_classes=number_class)  # 10 digits + 26 letters
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test

# Load and preprocess the data
folder_path = folder_train_path
X_train, X_test, y_train, y_test = load_and_preprocess_data(folder_path)

# Define the architecture of the AlexNet model
def create_alexnet():
    model = models.Sequential()
    
    # Convolutional layers
    model.add(layers.Conv2D(96, (11, 11), strides=(4, 4), activation='relu', input_shape=(227, 227, 3)))
    model.add(layers.MaxPooling2D((3, 3), strides=(2, 2)))
    model.add(layers.Conv2D(256, (5, 5), activation='relu'))
    model.add(layers.MaxPooling2D((3, 3), strides=(2, 2)))
    model.add(layers.Conv2D(384, (3, 3), activation='relu'))
    model.add(layers.Conv2D(384, (3, 3), activation='relu'))
    model.add(layers.Conv2D(256, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((3, 3), strides=(2, 2)))
    
    # Flatten layer
    model.add(layers.Flatten())
    
    # Fully connected layers
    model.add(layers.Dense(4096, activation='relu'))
    model.add(layers.Dense(4096, activation='relu'))
    model.add(layers.Dense(number_class, activation='softmax'))  # 10 digits + 20 letters
    
    return model

# Create the AlexNet model
model = create_alexnet()

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_split=0.2)



# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print('Test accuracy:', test_acc)

# Save the trained model
model.save('alexnet_model.h5')
