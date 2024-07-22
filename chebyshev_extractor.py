import numpy as np
from skimage.transform import resize
from skimage import color

def color_histogram_feature_extractor(image, bbox, target_size=(64, 128), bins=32):
    """
    Extract color histogram features from an image region defined by a bounding box.
    
    Args:
    image (numpy.ndarray): Full image (assumed to be in RGB format)
    bbox (tuple): Bounding box coordinates (x1, y1, x2, y2)
    target_size (tuple): Size to resize the cropped image to before feature extraction
    bins (int): Number of bins for each color channel in the histogram

    Returns:
    numpy.ndarray: Feature vector (color histogram)
    """
    # Crop the bounding box from the image
    x1, y1, x2, y2 = map(int, bbox)
    cropped = image[y1:y2, x1:x2]
    
    # Resize the cropped image to a standard size
    resized = resize(cropped, target_size, anti_aliasing=True)
    
    # Convert to LAB color space for better color representation
    lab_image = color.rgb2lab(resized)
    
    # Compute color histogram for each channel
    L_hist, _ = np.histogram(lab_image[:,:,0], bins=bins, range=(0, 100))
    A_hist, _ = np.histogram(lab_image[:,:,1], bins=bins, range=(-128, 127))
    B_hist, _ = np.histogram(lab_image[:,:,2], bins=bins, range=(-128, 127))
    
    # Concatenate histograms to form the feature vector
    features = np.concatenate((L_hist, A_hist, B_hist))
    
    # Normalize the feature vector
    features_normalized = features / np.sum(features)
    
    return features_normalized

def chebyshev_distance(hist1, hist2):
    """
    Compute the Chebyshev distance between two histograms.
    
    Args:
    hist1, hist2 (numpy.ndarray): Histograms to compare

    Returns:
    float: Chebyshev distance between the histograms
    """
    return np.max(np.abs(hist1 - hist2))

# Usage example:
# features1 = color_histogram_feature_extractor(frame1, bbox1)
# features2 = color_histogram_feature_extractor(frame2, bbox2)
# distance = chebyshev_distance(features1, features2)