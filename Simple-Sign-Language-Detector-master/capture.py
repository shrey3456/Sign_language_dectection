import cv2
import numpy as np
import os


def nothing(x):
    pass


image_x, image_y = 64, 64


def create_folder(folder_name):
    """Create folders for training and test datasets."""
    try:
        os.makedirs(f'./mydata/training_set/{folder_name}', exist_ok=True)
        os.makedirs(f'./mydata/test_set/{folder_name}', exist_ok=True)
    except Exception as e:
        print(f"Error creating directories: {e}")


def capture_images(ges_name, train_limit=350, test_limit=50):
    """Capture images for the specified gesture."""
    create_folder(ges_name)

    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("Error: Could not access the camera.")
        return

    cv2.namedWindow("Gesture Capture")
    cv2.namedWindow("Trackbars")

    cv2.createTrackbar("L - H", "Trackbars", 0, 179, nothing)
    cv2.createTrackbar("L - S", "Trackbars", 0, 255, nothing)
    cv2.createTrackbar("L - V", "Trackbars", 0, 255, nothing)
    cv2.createTrackbar("U - H", "Trackbars", 179, 179, nothing)
    cv2.createTrackbar("U - S", "Trackbars", 255, 255, nothing)
    cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)

    training_set_image_name = 1
    test_set_image_name = 1
    img_counter = 0

    while True:
        ret, frame = cam.read()
        if not ret:
            print("Failed to grab frame.")
            break

        frame = cv2.flip(frame, 1)
        l_h = cv2.getTrackbarPos("L - H", "Trackbars")
        l_s = cv2.getTrackbarPos("L - S", "Trackbars")
        l_v = cv2.getTrackbarPos("L - V", "Trackbars")
        u_h = cv2.getTrackbarPos("U - H", "Trackbars")
        u_s = cv2.getTrackbarPos("U - S", "Trackbars")
        u_v = cv2.getTrackbarPos("U - V", "Trackbars")

        lower_hsv = np.array([l_h, l_s, l_v])
        upper_hsv = np.array([u_h, u_s, u_v])

        cv2.rectangle(frame, (425, 100), (625, 300), (0, 255, 0), 2)
        roi = frame[102:298, 427:623]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_hsv, upper_hsv)

        cv2.putText(frame, f"Captured: {img_counter}", (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imshow("Gesture Capture", frame)
        cv2.imshow("Mask", mask)

        key = cv2.waitKey(1)
        if key == ord('c'):
            img_name = ""
            if training_set_image_name <= train_limit:
                img_name = f"./mydata/training_set/{ges_name}/{training_set_image_name}.png"
                training_set_image_name += 1
            elif test_set_image_name <= test_limit:
                img_name = f"./mydata/test_set/{ges_name}/{test_set_image_name}.png"
                test_set_image_name += 1

            if img_name:
                resized_img = cv2.resize(mask, (image_x, image_y))
                cv2.imwrite(img_name, resized_img)
                print(f"{img_name} written!")
                img_counter += 1

        elif key == 27:  # Escape key
            print("Exiting...")
            break

        if test_set_image_name > test_limit:
            print("Data collection complete!")
            break

    cam.release()
    cv2.destroyAllWindows()


# Main
gesture_name = input("Enter gesture name: ").strip()
capture_images(gesture_name)
