from PIL import Image
import numpy as np

def load_image(image_path):
    """
    Loads an image from the given file path using the PIL library.
    """
    image = Image.open(image_path)
    array = np.array(image)
    return array
