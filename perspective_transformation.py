import cv2
import numpy as np

# Load the image
image_path = 'path_to_your_image.jpg'
image = cv2.imread(image_path)

# Define source points (the points in your original image)
src_points = np.float32([
    [x1, y1],  # Top-left corner
    [x2, y2],  # Top-right corner
    [x3, y3],  # Bottom-right corner
    [x4, y4]   # Bottom-left corner
])

# Define destination points (how you want the points to be in the output image)
dst_points = np.float32([
    [0, 0],                               # Top-left corner
    [image.shape[1] - 1, 0],              # Top-right corner
    [image.shape[1] - 1, image.shape[0] - 1],  # Bottom-right corner
    [0, image.shape[0] - 1]               # Bottom-left corner
])

# Compute the perspective transform matrix
M = cv2.getPerspectiveTransform(src_points, dst_points)

# Apply the perspective transformation to the image
warped_image = cv2.warpPerspective(image, M, (image.shape[1], image.shape[0]))

# Display the original and warped images
cv2.imshow('Original Image', image)
cv2.imshow('Warped Image', warped_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
