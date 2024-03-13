import cv2
import numpy as np
from perspective_transformation import ImagePerspectiveTransformer
# from height_measurement_from_seg import VehicleTracker  # Uncomment and use as needed

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file {video_path}")
        return
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def main():
    image_path = 'transformation_image.jpg'  # Ensure this path is correct
    src_points = np.float32([[100, 100], [1100, 100], [1100, 800], [100, 800]])
    width, height = 644, 360
    dst_points = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
    ref_points = (np.array([50, 50]), np.array([150, 50]))
    real_length_cm = 100

    sample_image = cv2.imread(image_path)
    if sample_image is None:
        print(f"Failed to load image at path: {image_path}")
        return

    transformer = ImagePerspectiveTransformer(sample_image, src_points, dst_points, ref_points, real_length_cm)
    transformer.apply_perspective_transform()
    transformer.calculate_scale_factor()
    distance_cm = transformer.measure_distance(np.array([200, 200]), np.array([400, 400]))
    transformer.display_images()
    print(f"The real-world distance between the two points is {distance_cm} cm")

    video_path = 'passing_cars.mp4'  # Ensure this file exists in the root directory
    process_video(video_path)

if __name__ == "__main__":
    main()