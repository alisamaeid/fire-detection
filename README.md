# Fire Detection Using Machine Learning

This project focuses on detecting fire in images using two different machine learning approaches: K-Nearest Neighbors (KNN) and a Neural Network (NN) built with TensorFlow Keras. The dataset consists of images categorized as either "fire" or "non-fire," and two models are trained separately to classify them.

## Project Structure

```
├── test_images/                 # Sample images for testing models
├── fire_detection_with_KNN.py   # Script to train the KNN model
├── use_saved_KNN_model.py       # Script to test the trained KNN model
├── saved_knn_model.z            # Pre-trained KNN model
├── fire_detection_with_NN.py    # Script to train the Neural Network model
├── use_saved_NN_model.py        # Script to test the trained NN model
├── saved_NN_model.h5            # Pre-trained Neural Network model
├── requirements.txt             # List of dependencies for the project
├── environment.yml              # Conda environment configuration file
├── README.md                    # Project documentation
```
## Dataset

The dataset has two categories:
1. Fire Images 
2. Non-Fire Images
Each image is resized to 32x32 pixels, converted to a numpy array, and normalized before training.

Go to [Kaggle](https://www.kaggle.com/) to download the dataset
**Dataset Link:** [Fire Detection Dataset](https://www.kaggle.com/your-dataset-link)

## Models Overview

### K-Nearest Neighbors (KNN) Model
- The `fire_detection_with_KNN.py` script trains a KNN model on the dataset.
- Once trained, the model is saved as `saved_knn_model.z`.
- The `use_saved_KNN_model.py` script loads this model and classifies new images.
- Performance is acceptable but not as strong as the neural network model.

### Neural Network (NN) Model
- The `fire_detection_with_NN.py` script trains a neural network using TensorFlow Keras.
- Once trained, the model is saved as `saved_NN_model.h5`.
- The `use_saved_NN_model.py` script loads this model and classifies new images.
- This model has better accuracy and generalization compared to the KNN model.

## Running the Project

### 1. Setting Up the Environment
Ensure you have Python installed along with the required libraries. You can install dependencies using:
```bash
pip install -r requirements.txt
```
Alternatively, if you are using Conda, you can set up the environment using:
```bash
conda env create -f environment.yml
conda activate fire-detection
```

### 2. Training the Models
To train the KNN model:
```bash
python fire_detection_with_KNN.py
```
To train the Neural Network model:
```bash
python fire_detection_with_NN.py
```

### 3. Testing the Models
To test the KNN model:
```bash
python use_saved_KNN_model.py
```
To test the Neural Network model:
```bash
python use_saved_NN_model.py
```

## Contributing
Feel free to contribute by improving the models or optimizing the dataset handling. 

## References

- OpenCV: [https://opencv.org/](https://opencv.org/)
- Scikit-learn: [https://scikit-learn.org/](https://scikit-learn.org/)

## License
This project is open-source. You are free to use and modify it as needed.

