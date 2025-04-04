import cv2
import numpy as np
import os
import pandas as pd

def convex_hull(image_path):
    """
    Applies convex hull transformation to a binarized image of a hand gesture.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"File not found: {image_path}")
    _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    hull_image = np.zeros_like(binary)
    for contour in contours:
        hull = cv2.convexHull(contour)
        cv2.drawContours(hull_image, [hull], 0, 255, thickness=cv2.FILLED)

    cv2.imwrite("convex_hull.jpg", hull_image)
    return hull_image

def process_image(image_path, size=(64, 64)):
    """
    Loads an image, converts to grayscale, resizes, and flattens.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, size)  # Resize to fixed dimensions
    image = image.flatten().astype(np.float32)
    
    # Normalize to [0, 1]
    image = image/255
    
    return image

def create_df(input_root, output_csv):
    data = []
    labels = []

    for class_label in range(20):  # Class folders: 0 through 19
        class_folder = os.path.join(input_root, str(class_label))

        if not os.path.exists(class_folder):
            print(f"Skipping missing folder: {class_folder}")
            continue

        for file_name in os.listdir(class_folder):
            if file_name.lower().endswith((".jpg", ".png")):
                image_path = os.path.join(class_folder, file_name)
                processed_image = process_image(image_path)
                
                if processed_image is None:
                    continue
                
                labels.append(class_label)
                data.append(processed_image)


    df = pd.DataFrame(data)
    df.insert(0, "label", labels)
    df.to_csv(output_csv, index=False)

create_df("../data/test/", '../data/test_processed/test.csv')
create_df("../data/train/", '../data/test_processed/train.csv')
