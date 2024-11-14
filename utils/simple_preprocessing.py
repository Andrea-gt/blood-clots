from openslide import OpenSlide
import numpy as np
import cv2
import tensorflow as tf

def preprocess(image_path, resolution):
    slide = OpenSlide(image_path)
    region = (1000, 1000)    
    size = (5000, 5000)
    
    # Read the region from the slide
    image = slide.read_region(region, 0, size)
    
    # Convert the PIL image to a NumPy array
    image = np.array(image)
    
    # Convert from BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)

    # Resize the image
    image = tf.image.resize(image, (resolution, resolution))
    
    # Normalize pixel values to the range [0, 1]
    normalized_image = image / 255.0

    return normalized_image