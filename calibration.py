import cv2
import numpy as np
import logging
import os
import time
from glob import glob

class CameraCalibrationService:
    UPLOADS_DIR = "calibrations"  

    def __init__(self, checkerboard_dims, image_paths):
        self.calib_data_path = self.get_calibration_data_path()
        self.image_paths = image_paths
        self.checkerboard_dims = checkerboard_dims

    def get_calibration_data_path(self):
        calib_dir = os.path.join(self.UPLOADS_DIR)
        os.makedirs(calib_dir, exist_ok=True)
        return calib_dir

    def find_checkerboard_corners(self, images):
        objpoints = [] 
        imgpoints = []  
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        objp = np.zeros((self.checkerboard_dims[0] * self.checkerboard_dims[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.checkerboard_dims[0], 0:self.checkerboard_dims[1]].T.reshape(-1, 2)

        for img in images:
            try:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            except cv2.error as e:
                logging.error(f"OpenCV error in converting image to grayscale: {str(e)}")
                continue  

            ret, corners = cv2.findChessboardCorners(gray, (self.checkerboard_dims[0], self.checkerboard_dims[1]), None)
            if ret:
                objpoints.append(objp)
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners2)

        if not objpoints or not imgpoints:
            raise ValueError("Could not find checkerboard corners in any of the images.")

        return objpoints, imgpoints

    def calibrate_camera(self, objpoints, imgpoints, image_shape):
        try:
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, image_shape[::-1], None, None)
            if not ret:
                raise ValueError("Camera calibration failed.")
            return {"ret": ret, "mtx": mtx, "dist": dist, "rvecs": rvecs, "tvecs": tvecs}
        except cv2.error as e:
            logging.error(f"OpenCV error during camera calibration: {str(e)}")
            raise ValueError("Error during camera calibration.")

    def save_calibration_parameters(self, params):
        try:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = f"calibration_{timestamp}.npz"
            filepath = os.path.join(self.calib_data_path, filename)
           
            np.savez_compressed(
                filepath,
                cameraMatrix=params["mtx"],
                distCoeffs=params["dist"],
                rvecs=params["rvecs"],
                tvecs=params["tvecs"],
            )
            return filepath
        except Exception as e:
            logging.error(f"Error when saving calibration parameters: {str(e)}")
            raise ValueError("Error when saving calibration parameters.")  

    def perform_calibration(self, images):
        try:
            if not images:
                raise ValueError("No images provided. Please check the image paths.")
            objpoints, imgpoints = self.find_checkerboard_corners(images)
            calibration_params = self.calibrate_camera(objpoints, imgpoints, images[0].shape[:2])
            return self.save_calibration_parameters(calibration_params)
        except Exception as e:
            logging.error(f"Error during camera calibration process: {str(e)}")
            raise ValueError(f"Calibration failed: {str(e)}")

def load_calibration_images():
  
    image_paths = glob('./calibration_images/*.png')  # Example for JPEG images in a subdirectory called 'calibration_images'
    images = []
    for path in image_paths:
        img = cv2.imread(path)
        if img is not None:
            images.append(img)
        else:
            logging.error(f"Failed to load image at path: {path}")
    return images



def main():
    checkerboard_dims = (7, 9)  
    calibration_images = load_calibration_images()

    if not calibration_images:
        logging.error("No calibration images loaded. Please check your files.")
        return
    calibration_service = CameraCalibrationService(checkerboard_dims, None)
    calibration_result_path = calibration_service.perform_calibration(calibration_images)
    print(f"Calibration parameters saved to: {calibration_result_path}")

if __name__ == "__main__":
    main()
