import os
import cv2

def resize_images(directory):
    if not os.path.isdir(directory):
        print(f"Error: Directory '{directory}' not found.")
        return

    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            print(f"Resizing {filename}...")
            img = cv2.imread(filepath)
            resized_img = cv2.resize(img, (224, 224))
            cv2.imwrite(filepath, resized_img)

    print("All images resized successfully.")

# Set the directory path here
directory_path = "/home/vg245/SegVit/data/google_earth_sample"

resize_images(directory_path)