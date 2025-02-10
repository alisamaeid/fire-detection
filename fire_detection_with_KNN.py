
import cv2
from sklearn.model_selection import train_test_split
import glob # is used to find files and directories that match a specified pattern(you can use 'os' instead of 'glob')
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from joblib import dump,load # use dump to save a model , use load to load it later
from sklearn.metrics import recall_score, precision_score,f1_score,accuracy_score


def load_all_data():

    """ load and preprocess the data (resizing, normalizing, flattening), split train and test subsets """
    
    dataList = []
    Labels = []
    for i,address in enumerate(glob.glob("fire_dataset\\*\\*")): # read the images' addresses 

        img = cv2.imread(address) # use image address to read the image
        img = cv2.resize(img,(32,32)) # resize images 32x32
        img = img/255.0 # normalize images
        img = img.flatten() # convert a multi-dimensional array into a 1D array.
        dataList.append(img) # add the image to the dataList
        #print(img.shape)
        label = address.split("\\")[-1].split(".")[0] # use first part of each image for the label of it(fire, non-fire)
        Labels.append(label)

        if i % 100 == 0:
            print(f"{i}/{1000} of the images was processed") # see the peogress of the loading the data

    global data_array # using global to use data_array out side of the function
    data_array = np.array(dataList) # convert data list to a numpy array

    X_train, X_test, y_train, y_test = train_test_split(data_array, Labels, test_size=0.2, random_state=2) # split train and test subsets

    return  X_train, X_test, y_train, y_test


def KNN_algorithm():
    """ Train model with KNN algorithm"""
    # change the algorithm or tune hyperparameters To achieve better results: 
    clf = KNeighborsClassifier(n_neighbors=5) #use KNN to train so we should have converted our data to a numpy array which is done in the function load_data 
    clf.fit(X_train,y_train)

    return clf

def show_train_results():
    """
    predict on test set, calculate accuracy_score, precision_score, recall_score, f1_score
    """
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test,y_pred) * 100
    precision = precision_score(y_test,y_pred, pos_label="fire")* 100
    recall = recall_score(y_test,y_pred,pos_label="fire") * 100
    f1 = f1_score(y_test,y_pred,pos_label="fire") * 100

    print(f"accuracy : {accuracy:.2f}")
    print(f"precision : {precision:.2f}")
    print(f"recall : {recall:.2f}")
    print(f"f1 : {f1:.2f}")


print("loading the data ...")
X_train, X_test, y_train, y_test = load_all_data() # load data(resizing,normalizing,flattening)

print("-----------------------------------")
print(f"shape of the dataset : {data_array.shape}\nthere is {data_array.shape[0]} images\neach one has {data_array.shape[1]} features.")
print("-----------------------------------")

print("Start training ...")
clf = KNN_algorithm()

print("*** The training is finished. See the results below: ")
show_train_results()

dump(clf,"saved_knn_model.z") # save model to use it later in use_saved_model.py script
