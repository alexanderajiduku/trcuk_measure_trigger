import cv2
import logging




def load_images(image_paths):
    images = []
    for path in image_paths:
        img = cv2.imread(path)
        if img is not None:
            images.append(img)
        else:
            logging.error(f"Failed to load image at path: {path}")
    return images
