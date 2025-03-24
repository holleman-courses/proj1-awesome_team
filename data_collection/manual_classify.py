import numpy as np
import cv2
import os

pencil_counter = 421
no_pencil_counter = 421

parent_dir = r"D:\School\Intro IOT\proj1-awesome_team"
images_dir = r'data_collection\images'
start_dir = os.path.join(parent_dir, images_dir)

pencil_path = os.path.join(parent_dir, r"dataset\pencil")
no_pencil_path = os.path.join(parent_dir, r"dataset\no_pencil")

os.makedirs(pencil_path, exist_ok=True)
os.makedirs(no_pencil_path, exist_ok=True)

def classify_image(file_path):
    global pencil_counter, no_pencil_counter

    image = cv2.imread(file_path)
    cv2.imshow("Image", image)  # Show image in a lightweight OpenCV window
    key = cv2.waitKey(0) & 0xFF  # Wait for a key press (no need to press Enter)

    path = None
    count = 0
    label = None

    if key == ord('d'):
        cv2.destroyAllWindows()
        return  # Skip this image
    elif key == ord('a'):
        label = 'no_pencil'
        path = no_pencil_path
        count = no_pencil_counter
        no_pencil_counter += 1
    elif key == ord('s'):
        label = 'pencil'
        path = pencil_path
        count = pencil_counter
        pencil_counter += 1

    if path:
        save_path = os.path.join(path, f"{label}_{count}.png")
        cv2.imwrite(save_path, image)
        print(f"Saved to {save_path}")

    cv2.destroyAllWindows()  # Close the OpenCV window after classification

for filename in os.listdir(start_dir):
    file_path = os.path.join(start_dir, filename)
    if os.path.isfile(file_path):
        print(f"Processing file: {file_path}")
        classify_image(file_path)
