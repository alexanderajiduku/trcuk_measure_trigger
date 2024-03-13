import cv2
import numpy as np

class ImagePerspectiveTransformer:
    def __init__(self, image, src_points, dst_points, ref_points, real_length_cm):
        self.image = image
        self.src_points = src_points
        self.dst_points = dst_points
        self.ref_point1, self.ref_point2 = ref_points
        self.real_length_cm = real_length_cm
        self.warped_image = None
        self.scale_factor = None

    def apply_perspective_transform(self):
        M = cv2.getPerspectiveTransform(self.src_points, self.dst_points)
        self.warped_image = cv2.warpPerspective(self.image, M, (self.image.shape[1], self.image.shape[0]))

    def calculate_scale_factor(self):
        pixel_distance = np.linalg.norm(self.ref_point1 - self.ref_point2)
        self.scale_factor = self.real_length_cm / pixel_distance

    def measure_distance(self, point1, point2):
        if self.scale_factor is None:
            raise ValueError("Scale factor not calculated. Please call calculate_scale_factor() first.")
        pixel_distance = np.linalg.norm(point1 - point2)
        return pixel_distance * self.scale_factor

    def display_images(self):
        display_original = cv2.resize(self.image, (self.image.shape[1] // 3, self.image.shape[0] // 3))
        display_warped = cv2.resize(self.warped_image, (self.warped_image.shape[1] // 3, self.warped_image.shape[0] // 3))
        cv2.imshow('Original Image', display_original)
        cv2.imshow('Warped Image', display_warped)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

