import cv2
import numpy as np
from ultralytics import YOLO
from yolo_segmentation import YOLOSegmentation
import pandas as pd
import os
from tracker import Tracker

class VehicleTracker:
    def __init__(self, video_path, model_path, output_path='output.avi', display_duration=60):
        self.video_path = video_path
        self.detection_model = YOLO(model_path)
        self.segmentation_model = YOLOSegmentation(model_path)
        self.output_path = output_path
        self.display_duration = display_duration
        self.vehicle_process = {}
        self.vehicle_display_info = {}
        self.tracker = Tracker()
        self.setup_output_directory('detected_frames')

        self.class_list = [
    "person", "bicycle", "car", "motorcycle", "airplane",
    "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
    "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
    "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife",
    "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed",
    "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

        # Drawing lines configuration
        self.red_line_x = 198
        self.blue_line_x = 868
        self.center_line_x = (self.blue_line_x + self.red_line_x) // 2
        self.offset = 6

    def setup_output_directory(self, directory_name):
        if not os.path.exists(directory_name):
            os.makedirs(directory_name)

    def process_video(self):
        cap = cv2.VideoCapture(self.video_path)
        ret, frame = cap.read()
        if not ret:
            print("Failed to read video")
            return

        # Setup video writer
        out = cv2.VideoWriter(self.output_path, cv2.VideoWriter_fourcc(*'XVID'), 20.0, (1020, 500))

        while ret:
            frame = cv2.resize(frame, (1020, 500))
            self.process_frame(frame, out)
            ret, frame = cap.read()

        # Release resources
        cap.release()
        out.release()
        cv2.destroyAllWindows()

    def process_frame(self, frame, out):
        results = self.detection_model.predict(frame)
        boxes = results[0].boxes.data.detach().cpu().numpy()
        detections = pd.DataFrame(boxes).astype("float")

        detected_cars = [row[:4].astype(int).tolist() for index, row in detections.iterrows() if self.class_list[int(row[5])] == 'car']
        bbox_id = self.tracker.update(detected_cars)

        self.handle_detections(frame, bbox_id)
        self.draw_lines(frame)
        self.display_vehicle_info(frame)

        out.write(frame)
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return

    def handle_detections(self, frame, bbox_id):
        for bbox in bbox_id:
            x3, y3, x4, y4, id = bbox
            cx = int((x3 + x4) / 2)
            if self.center_line_x - self.offset < cx < self.center_line_x + self.offset and id not in self.vehicle_process:
                self.vehicle_process[id] = 'crossed'
                frame_path = f'detected_frames/vehicle_{id}_center_line.jpg'
                cv2.imwrite(frame_path, frame)
                seg_img = cv2.imread(frame_path)
                if seg_img is not None:
                    seg_img = cv2.resize(seg_img, None, fx=0.7, fy=0.7)
                    _, _, seg_contours, _ = self.segmentation_model.detect(seg_img)
                    for seg in seg_contours:
                        y_coords = seg[:, 1] 
                        min_y = np.min(y_coords)
                        max_y = np.max(y_coords)
                        vertical_extent = max_y - min_y 

                        self.vehicle_display_info[id] = {
                            'vertical_extent': vertical_extent,
                            'display_frame_count': self.display_duration,
                            'position': (x4 + 10, y3 + 20)  
                        }

    def draw_lines(self, frame):
        cv2.line(frame, (self.red_line_x, 0), (self.red_line_x, frame.shape[0]), (0, 0, 255), 2)
        cv2.line(frame, (self.blue_line_x, 0), (self.blue_line_x, frame.shape[0]), (255, 0, 0), 2)
        cv2.line(frame, (self.center_line_x, 0), (self.center_line_x, frame.shape[0]), (0, 255, 0), 2)

    def display_vehicle_info(self, frame):
        completed_vehicles = len([v for v in self.vehicle_process.values() if v == 'crossed'])
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        color = (0, 255, 0)  # Green color for text
        background_color = (0, 0, 0)  # Black background for better readability

        # Display the count of completed vehicles on the top-left corner
        text = f'Completed Vehicles: {completed_vehicles}'
        text_offset_x, text_offset_y = 10, 20
        cv2.rectangle(frame, (text_offset_x, text_offset_y - 15), (text_offset_x + 230, text_offset_y + 5), background_color, -1)
        cv2.putText(frame, text, (text_offset_x, text_offset_y), font, font_scale, color, thickness)

        # Display vehicle info for each detected vehicle
        for vehicle_id, info in self.vehicle_display_info.items():
            if info['display_frame_count'] > 0:
                text = f'Vehicle ID {vehicle_id} Height: {info["vertical_extent"]:.2f} px'
                position = info['position']
                cv2.rectangle(frame, (position[0] - 2, position[1] - 15), (position[0] + 200, position[1] + 5), background_color, -1)
                cv2.putText(frame, text, position, font, font_scale, color, thickness)
                info['display_frame_count'] -= 1  # Decrement the display frame count for timed display

