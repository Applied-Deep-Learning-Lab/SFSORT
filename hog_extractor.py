import numpy as np
from skimage.feature import hog
from skimage.transform import resize

def simple_feature_extractor(image, bbox, target_size=(64, 128)):
    """
    Extract features from an image region defined by a bounding box.
    
    Args:
    image (numpy.ndarray): Full image
    bbox (tuple): Bounding box coordinates (x1, y1, x2, y2)
    target_size (tuple): Size to resize the cropped image to before feature extraction

    Returns:
    numpy.ndarray: Feature vector
    """
    # Crop the bounding box from the image
    x1, y1, x2, y2 = map(int, bbox)
    cropped = image[y1:y2, x1:x2]
    
    # Resize the cropped image to a standard size
    resized = resize(cropped, target_size, anti_aliasing=True)
    
    # Convert to grayscale if the image is in color
    if resized.ndim == 3:
        resized = np.mean(resized, axis=2)
    
    # Extract HOG features
    features, _ = hog(resized, orientations=9, pixels_per_cell=(8, 8), 
                      cells_per_block=(2, 2), visualize=True)
    
    # Normalize the feature vector
    features_normalized = features / np.linalg.norm(features)
    
    return features_normalized

# Usage example:
# features = simple_feature_extractor(frame, bbox)