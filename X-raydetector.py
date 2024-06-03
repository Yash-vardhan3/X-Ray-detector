import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from tkinter import filedialog
from tkinter import Tk

# Step 1: Data Preparation
data_dir = r"C:\Users\HP\Downloads\archive\chest_xray\test"
classes = ["NORMAL", "PNEUMONIA"]
images = []
labels = []

for class_idx, class_name in enumerate(classes):
    for image_name in os.listdir(os.path.join(data_dir, class_name)):
        image = cv2.imread(os.path.join(data_dir, class_name, image_name))
        image = cv2.resize(image, (224, 224))  # Resize image to 224x224
        image = image.astype('float32') / 255.0  # Normalize pixel values
        images.append(image)
        labels.append(class_idx)

# Convert lists to numpy arrays
images = np.array(images)
labels = np.array(labels)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Convert labels to one-hot encoding
y_train = to_categorical(y_train, num_classes=2)
y_test = to_categorical(y_test, num_classes=2)

# Step 2: Model Building
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(2, activation='softmax')  # Output layer with 2 neurons (Normal, Pneumonia)
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Step 3: Model Training
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Step 4: Model Evaluation
loss, accuracy = model.evaluate(X_test, y_test)
print("Test Accuracy:", accuracy)

# Step 5: Model Saving
model.save("medical_image_classifier.h5")

# Step 6: Prediction
def predict_real_life_image(image_path):
    # Load the trained model
    model = load_model("medical_image_classifier.h5")
    
    # Preprocess the image
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))  # Resize image to 224x224
    image = image.astype('float32') / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    
    # Make prediction
    predictions = model.predict(image)
    
    # Interpret prediction
    if predictions[0][0] > predictions[0][1]:
        return "Normal"
    else:
        return "Pneumonia"

# Function to browse for an image file
def browse_image():
    Tk().withdraw() # to prevent root window from appearing
    image_path = filedialog.askopenfilename() # show an "Open" dialog box and return the path to the selected file
    prediction = predict_real_life_image(image_path)
    print("Prediction:", prediction)

# Step 7: Visualization
# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Example usage
browse_image()
