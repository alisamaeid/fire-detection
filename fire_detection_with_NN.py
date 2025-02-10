import os  
# Disable OneDNN optimizations for TensorFlow (can sometimes affect performance or compatibility)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import matplotlib.pyplot as plt  # plotting 
import cv2  # OpenCV library for image processing
from sklearn.model_selection import train_test_split  # For splitting dataset into training and testing sets
import glob  # For reading file paths
import numpy as np  
from sklearn.preprocessing import LabelEncoder  # For encoding labels into numerical values
from tensorflow.keras.utils import to_categorical  # type: ignore # Converts labels into one-hot encoding
from tensorflow.keras import models, layers  # type: ignore # To define and build neural networks

def load_all_data():
    """
    load_all_data():
    Preprocessing the dataset [reading, resizing, normalizing, flattening, splitting, encoding]
    """

    dataList = []  # store processed images
    Labels = []  # store image labels

    # Iterate through all image file paths in the "fire_dataset" directory
    for i, address in enumerate(glob.glob("fire_dataset\\*\\*")):  

        img = cv2.imread(address)  # Read the image 
        img = cv2.resize(img, (32, 32))  # Resize image to 32x32 
        img = img / 255.0  # Normalize pixel values (scale between 0 and 1)
        img = img.flatten()  # Flatten 3D image array into 1D array
        dataList.append(img)  # Add processed image to data list

        # Extract label from the filename (fire or non-fire)
        label = address.split("\\")[-1].split(".")[0]  
        Labels.append(label)  

        if i % 100 == 0:
            print(f"{i}/{1000} of the images were processed")  

    
    data_array = np.array(dataList)  # Convert data list into a NumPy array for efficient processing
    print(f"Shape of the dataset: {data_array.shape}\nThere are {data_array.shape[0]} images\neach one has {data_array.shape[1]} features.")

    X_train, X_test, y_train, y_test = train_test_split(data_array, Labels, test_size=0.2, random_state=2) # Split dataset

    # Encode labels into numerical values (fire = 1, non-fire = 0)
    enc = LabelEncoder()
    y_train = enc.fit_transform(y_train)
    y_test = enc.transform(y_test)

    # Convert labels to one-hot encoded format
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    return X_train, X_test, y_train, y_test  

def my_neural_network(save=False):
    """
    my_neural_network():
    Creates the architecture of the neural network, trains, and evaluates it.
    To save the model, set save=True.
    """

    model = models.Sequential([
        layers.Dense(20, activation="relu", input_dim=3072),  # First hidden layer with 20 neurons, ReLU activation
        layers.Dense(8, activation="relu"),  # Second hidden layer with 8 neurons, ReLU activation
        layers.Dense(2, activation="softmax")  # Output layer with 2 neurons (fire/non-fire), softmax for classification
    ])

    model.summary()  # Print the model architecture

    # Compile the model with SGD optimizer and categorical cross-entropy loss
    model.compile(optimizer="sgd", loss="categorical_crossentropy", metrics=["accuracy"])

    # Train the model using training data and validate with test data
    H = model.fit(X_train, y_train, batch_size=32, epochs=100, validation_data=(X_test, y_test))

    # Evaluate the model on the test dataset
    loss, acc = model.evaluate(X_test, y_test)
    print(f"Loss: {loss:.2f}, Accuracy: {acc:.2f}")

    # Save the model if save=True
    if save:
        model.save("saved_NN_model.h5")

    return H  # Return training history

def show_results():
    """
    show_results():
    Displays training and testing loss and accuracy plots.
    """

    plt.style.use("ggplot")  # Set plot style

    # Plot accuracy and loss values over epochs
    plt.plot(H.history["accuracy"], label="train accuracy")
    plt.plot(H.history["val_accuracy"], label="test accuracy")
    plt.plot(H.history["loss"], label="train loss")
    plt.plot(H.history["val_loss"], label="test loss")

    plt.legend()  
    plt.xlabel("Epochs")  
    plt.ylabel("Loss") 
    plt.title("Training & Evaluating on Fire Detection Dataset")  
    plt.show()  

# Print information about the functions
print("-----------------------------------")
print(load_all_data.__doc__)  
print(my_neural_network.__doc__)  
print(show_results.__doc__)  
print("-----------------------------------")

print("loading data ...")
# Load dataset and preprocess it
X_train, X_test, y_train, y_test = load_all_data()  

print("start training ...")
# Train and evaluate the neural network
H = my_neural_network(save=True)

print("See the results on the chart")
# Show the training and evaluation results
show_results()

print("** Finished **") 
