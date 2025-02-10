import glob
import cv2
from joblib import load
import numpy as np

# remember to change the saved model directory!
clf = load("saved_knn_model.z") # load pre_trained model

for test_item in glob.glob("test_images\\*"): # load test images one by one
    # normalize test images as well.
    img = cv2.imread(test_item) # read image
    r_img = cv2.resize(img,(32,32)) # resize image 32x32
    r_img = r_img/255.0 # normalize image
    r_img = r_img.flatten() # falatten image

    r_img = np.array([r_img]) # convert images to 2D arrays to use predict method
    label = clf.predict(r_img)[0] # model prediction on iamge

    cv2.putText(img, label, (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1.9, (0, 255, 0), 2) # put a label on the top left of the image
    cv2.imshow("image", img) # show the image 
    cv2.waitKey(0) # wait for a click to show the next image

cv2.destroyAllWindows() # close the image window
