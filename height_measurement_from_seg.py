import cv2
import numpy as np
from ultralytics import YOLO
from yolo_segmentation import YOLOSegmentation
import pandas as pd
from tracker import Tracker
import os

detection_model = YOLO('yolov8n-seg.pt')
segmentation_model = YOLOSegmentation('yolov8n-seg.pt')
if not os.path.exists('detected_frames'):
    os.makedirs('detected_frames')

class_list = [
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

cap = cv2.VideoCapture('passing_cars.mp4')
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (1020, 500))

red_line_x = 198
blue_line_x = 868
center_line_x = (blue_line_x + red_line_x) // 2
offset = 6
tracker = Tracker()
vehicle_process = {}
vehicle_display_info = {}
display_duration = 60 

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (1020, 500))
    results = detection_model.predict(frame)
    boxes = results[0].boxes.data.detach().cpu().numpy()
    detections = pd.DataFrame(boxes).astype("float")

    detected_cars = [row[:4].astype(int).tolist() for index, row in detections.iterrows() if class_list[int(row[5])] == 'car']
    bbox_id = tracker.update(detected_cars)

    for bbox in bbox_id:
        x3, y3, x4, y4, id = bbox
        cx = int((x3 + x4) / 2)
        if center_line_x - offset < cx < center_line_x + offset and id not in vehicle_process:
            vehicle_process[id] = 'crossed'
            frame_path = f'detected_frames/vehicle_{id}_center_line.jpg'
            cv2.imwrite(frame_path, frame)
            seg_img = cv2.imread(frame_path)
            if seg_img is not None:
                seg_img = cv2.resize(seg_img, None, fx=0.7, fy=0.7)
                _, _, seg_contours, _ = segmentation_model.detect(seg_img)
                for seg in seg_contours:
                    y_coords = seg[:, 1]  # Extract all y-coordinates from the segmented contour
                    min_y = np.min(y_coords)
                    max_y = np.max(y_coords)
                    vertical_extent = max_y - min_y  # Calculate the vertical extent

                    # Update vehicle display info
                    vehicle_display_info[id] = {
                        'vertical_extent': vertical_extent,
                        'display_frame_count': display_duration,
                        'position': (x4 + 10, y3 + 20)  # Adjust as needed
                    }

    # Draw lines and text on the frame
    cv2.line(frame, (red_line_x, 0), (red_line_x, frame.shape[0]), (0, 0, 255), 2)
    cv2.line(frame, (blue_line_x, 0), (blue_line_x, frame.shape[0]), (255, 0, 0), 2)
    cv2.line(frame, (center_line_x, 0), (center_line_x, frame.shape[0]), (0, 255, 0), 2)
    completed_vehicles = len([v for v in vehicle_process.values() if v == 'crossed'])
    
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    color = (0, 255, 0)  
    background_color = (0, 0, 0)  


    text = f'Completed Vehicles - {completed_vehicles}'
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)

    text_offset_x = 10
    text_offset_y = 30
    background_start = (text_offset_x, text_offset_y - text_height - 4)
    background_end = (text_offset_x + text_width, text_offset_y + 4)

    cv2.rectangle(frame, background_start, background_end, background_color, -1)


    cv2.putText(frame, text, (text_offset_x, text_offset_y), font, font_scale, color, thickness)


    for vehicle_id, info in vehicle_display_info.items():
        if info['display_frame_count'] > 0:
            text = f'Vehicle Height: {info["vertical_extent"]}'
            (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
            text_offset_x = info['position'][0]
            text_offset_y = info['position'][1]

           
            cv2.rectangle(frame, (text_offset_x, text_offset_y - text_height - 4), (text_offset_x + text_width, text_offset_y + 4), background_color, -1)
            cv2.putText(frame, text, (text_offset_x, text_offset_y), font, font_scale, color, thickness)

            info['display_frame_count'] -= 1 

    out.write(frame)
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
