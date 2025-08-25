import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
import keras_tuner as kt
import shutil
import matplotlib.pyplot as plt
import seaborn as sns

# Print the current working directory
print(f"Current working directory: {os.getcwd()}")

# Path to the tuner directory
tuner_dir = r'C:\\Users\\asqua\\Desktop\\Anisha\\5th_Sem\\MiniProject\\my_dir'

# Remove the existing tuner directory if it exists
if os.path.exists(tuner_dir):
    shutil.rmtree(tuner_dir)

# Ensure the directory is created
if not os.path.exists(tuner_dir):
    os.makedirs(tuner_dir)
    print(f"Created directory: {tuner_dir}")

# Load and preprocess the data
def load_data(data_dir):
    images = []
    labels = []
    for label in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, label)
        if os.path.isdir(class_dir):  # Ensure it's a directory
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                img = cv2.imread(img_path)
                if img is not None:  # Ensure the image was read successfully
                    img = cv2.resize(img, (128, 128))
                    images.append(img)
                    labels.append(label)
    print(f"Loaded {len(images)} images and {len(labels)} labels")  # Debug statement
    return np.array(images), np.array(labels)

# Path to your dataset
data_dir = r'C:\\Users\\asqua\\Desktop\\Anisha\\5th_Sem\\MiniProject\\Oral Cancer Dataset'

# Load dataset
images, labels = load_data(data_dir)

# Encode labels
le = LabelEncoder()
labels = le.fit_transform(labels)

# Normalize images
images = images / 255.0

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

# Define the model creation function for hyperparameter tuning with transfer learning
def build_finetuned_model(hp):
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
    # Unfreeze some layers for fine-tuning
    for layer in base_model.layers[-4:]:
        layer.trainable = True

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(hp.Choice('dense_units', [128, 256, 512]), activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    # Try different optimizers
    optimizer = hp.Choice('optimizer', ['adam', 'rmsprop', 'sgd'])
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Combine Hyperband settings with more epochs
tuner = kt.Hyperband(
    build_finetuned_model,
    objective='val_accuracy',
    max_epochs=30,
    factor=3,
    directory=tuner_dir,
    project_name='oral_cancer_finetuning_v3'
)

# Perform hyperparameter tuning
tuner.search(datagen.flow(X_train, y_train, batch_size=32), epochs=30, validation_data=(X_test, y_test))

# Get the best model
best_model = tuner.get_best_models(num_models=1)[0]

# Evaluate the best model
loss, accuracy = best_model.evaluate(X_test, y_test)
print(f"Test accuracy: {accuracy}")

# Save the best model
best_model.save('best_oral_cancer_detector_finetuned_v3.keras')

# Generate confusion matrix and classification report
y_pred = (best_model.predict(X_test) >= 0.6).astype(int).flatten()
y_true = y_test

conf_matrix = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

class_report = classification_report(y_true, y_pred)
print("Classification Report:")
print(class_report)

# Plot confusion matrix
sns.heatmap(conf_matrix, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Predict on new images
def predict_image(image_path, threshold=0.4):
    if not os.path.isfile(image_path):
        return f"File not found: {image_path}"
    
    img = cv2.imread(image_path)
    if img is None:
        return f"Failed to read image: {image_path}"
    
    img = cv2.resize(img, (128, 128))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    prediction = best_model.predict(img)
    prediction_value = prediction[0][0]  # Extracting the prediction value
    print(f"Prediction value: {prediction_value}")  # Debugging print statement
    if prediction_value < threshold:
        return "No Cancer"
    else:
        return "Cancer"

# Example usage with adjusted threshold
example_image_path = r'C:\\Users\\asqua\\Desktop\\Anisha\\5th_Sem\\MiniProject\\Oral Cancer Dataset\\NON CANCER\\105.jpeg'
print(predict_image(example_image_path, threshold=0.4))
