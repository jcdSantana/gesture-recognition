import cv2
import numpy as np
import os
import pandas as pd
from sklearn.decomposition import TruncatedSVD
def convex_hull(image_path):
    """
    Applies convex hull transformation to a binarized image of a hand gesture.

    Parameters:
    image_path (str): Path to the input image.

    Returns:
    hull_image (numpy.ndarray): Processed image with the convex hull applied.
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
    
    Parameters:
    image_path (str): Path to the input image.\
    image size (int, int): Size of output image

    Returns:
    Processed image
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, size)  # Resize to fixed dimensions
    image = image.flatten()  # Convert 2D array to 1D vector
    return image


def apply_svd(df, n_components=100):
    """
    Applies Singular Value Decomposition (SVD) to reduce the dimensionality of a dataset.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing the dataset.
    n_components (int): Number of components to keep (default: 100).
    
    Returns:
    pd.DataFrame: Reduced dataset with SVD applied.
    """
   
    # Apply Truncated SVD for dimensionality reduction
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    reduced_features = svd.fit_transform(df)
    
    # Convert reduced data back to DataFrame and add labels
    reduced_df = pd.DataFrame(reduced_features, columns=[f"svd_{i}" for i in range(n_components)])
    
    return reduced_df

input_root = "../data/test/"
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
            
            labels.append(class_label)  # Store class label
            data.append(processed_image)  # Store processed image vector

# Convert to DataFrame
df = pd.DataFrame(data)
df = apply_svd(df, n_components=100)
df.insert(0, "label", labels)  

# Save to CSV
output_csv = '../data/test_processed/test.csv'
df.to_csv(output_csv, index=False)
print(f"Processing complete! CSV saved at {output_csv}")

