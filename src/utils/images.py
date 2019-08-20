"""Images utilities functions"""

import cv2
import numpy as np

def crop_box(img, box, offset=0.05):
    """Crop box"""

    shape = img.shape
    min_y = int(((box[0] - offset) if box[0] > offset else 0) * shape[0])
    min_x = int(((box[1] - offset) if box[1] > offset else 0) * shape[1])
    max_y = int(((box[2] + offset) if box[2] + offset < 1 else 1) * shape[0])
    max_x = int(((box[3] + offset) if box[3] + offset < 1 else 1) * shape[1])
    return img[int(min_y):int(max_y), int(min_x):int(max_x)]

def resize(img, size):
    """Resizes a given image"""

    new_width, new_height = size
    height, width, channels = img.shape
    ratio = min(new_width/width, new_height/height)
    new_x = int(width * ratio)
    new_y = int(height * ratio)
    resized_img = cv2.resize(img, (new_x, new_y))
    new_img = np.zeros((new_width, new_height, 3))
    x_offset = int((new_width - new_x) / 2)
    y_offset = int((new_height - new_y) / 2)
    new_img[y_offset:y_offset+new_y, x_offset:x_offset+new_x] = resized_img

    return new_img
