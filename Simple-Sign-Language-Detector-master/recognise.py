import cv2
import numpy as np
from keras.models import load_model
import time

# Load the trained model
classifier = load_model('Trained_model.h5')

def nothing(x):
    pass

# Define the target image size
image_x, image_y = 64, 64

# Prediction function
def predictor(image_path):
    from keras.preprocessing import image
    test_image = image.load_img(image_path, target_size=(64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = classifier.predict(test_image)

    # Get the predicted class with highest probability
    predicted_class = np.argmax(result, axis=1)[0]
    
    # Map the predicted class to its corresponding gesture
    gesture_map = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    
    return gesture_map[predicted_class]

# Set up the webcam
cam = cv2.VideoCapture(0)

# Create trackbars to adjust HSV range for hand detection
cv2.namedWindow("Trackbars")
cv2.createTrackbar("L - H", "Trackbars", 0, 179, nothing)
cv2.createTrackbar("L - S", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("L - V", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("U - H", "Trackbars", 179, 179, nothing)
cv2.createTrackbar("U - S", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)

# Window for displaying the webcam feed
cv2.namedWindow("Trackbars")


img_counter = 0
img_text = ''

while True:
    # Capture frame from webcam
    ret, frame = cam.read()
    frame = cv2.flip(frame, 1)  # Flip the frame horizontally

    # Get HSV trackbar values
    l_h = cv2.getTrackbarPos("L - H", "Trackbars")
    l_s = cv2.getTrackbarPos("L - S", "Trackbars")
    l_v = cv2.getTrackbarPos("L - V", "Trackbars")
    u_h = cv2.getTrackbarPos("U - H", "Trackbars")
    u_s = cv2.getTrackbarPos("U - S", "Trackbars")
    u_v = cv2.getTrackbarPos("U - V", "Trackbars")

    # Define the region of interest (ROI) for hand gesture
    img = cv2.rectangle(frame, (425, 100), (625, 300), (0, 255, 0), thickness=2)
    
    lower_blue = np.array([l_h, l_s, l_v])
    upper_blue = np.array([u_h, u_s, u_v])
    imcrop = frame[102:298, 427:623]
    
    # Convert to HSV and apply the mask
    hsv = cv2.cvtColor(imcrop, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
    # Resize and save the image for prediction
    img_name = f"hand_gesture_{time.time()}.png"
    save_img = cv2.resize(mask, (image_x, image_y))
    cv2.imwrite(img_name, save_img)
    print(f"{img_name} saved!")

    # Get the prediction result
    img_text = predictor(img_name)
    
    # Display the prediction text on the frame
    cv2.putText(frame, img_text, (30, 400), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (0, 255, 0))
    
    # Show the frames
    cv2.imshow("test", frame)
    cv2.imshow("mask", mask)

    # Break the loop if 'Esc' key is pressed
    if cv2.waitKey(1) == 27:
        break

# Release the webcam and close all windows
cam.release()
cv2.destroyAllWindows()
