
import cv2  # OpenCV for image processing
import glob  # For reading file paths
import numpy as np  
from tensorflow.keras.models import load_model  # type: ignore # To load a pre-trained model

# Load the trained neural network model for fire detection
clf = load_model("saved_NN_model.h5")  

# Define the output labels corresponding to the model’s predictions
output_label = ["fire", "non fire"]  
print("start predicting ...")
# Loop through all test images in the "test_images" folder to preprocess them and make predictions
for test_item in glob.glob("test_images\\*"):  
    
    img = cv2.imread(test_item)  # Read the image
    r_img = cv2.resize(img, (32, 32))  # Resize image to 32x32 (same size used during training)
    r_img = r_img / 255.0  # Normalize pixel values
    r_img = r_img.flatten()  # Flatten the image 

    r_img = np.array([r_img])  # Convert image into a 2D array for model prediction

    pred_out = clf.predict(r_img)[0]  # Get model predictions 
    max_pred = np.argmax(pred_out)  # Get the index of the highest probability prediction
    output = output_label[max_pred]  # Convert index to corresponding label ("fire" or "non fire")

    try:
        if output == "fire": # If the prediction is "fire", overlay red text on the image
            cv2.putText(img, "fire", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.9, (0, 0, 255), 2)  
            
        else: # If the prediction is "non fire", overlay green text on the image
            cv2.putText(img, "non fire", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.9, (0, 255, 0), 2)  

        cv2.imshow("image", img)  # Display the image with prediction
        cv2.waitKey(0)  # Wait for a key press to show the next image

    except:
        print("Unknown Error!!")  # Handle potential errors

# Close all OpenCV image windows after processing
cv2.destroyAllWindows()  

print("Done 👍!")