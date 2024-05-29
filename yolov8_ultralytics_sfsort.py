import random
import time
import cv2
import numpy as np
from ultralytics import YOLO
from SFSORT import SFSORT

# Model loading
session = YOLO('yolov8m.pt', task='detect')

def remove_stale_keys(data_dict, heartbeat_dict, max_age_seconds):
    now = time.time()
    for key, last_update_time in list(heartbeat_dict.items()):
        if now - last_update_time > max_age_seconds:
            del data_dict[key]
            del heartbeat_dict[key]

# All classes
names = session.names

# Load the video file
cap = cv2.VideoCapture('excavator.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the MP4 codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))

# Organize tracker arguments into standard format
tracker_arguments = {"dynamic_tuning": True, "cth": 0.7,
                     "high_th": 0.7, "high_th_m": 0.1,
                     "match_th_first": 0.6, "match_th_first_m": 0.05,
                     "match_th_second": 0.4, "low_th": 0.2,
                     "new_track_th": 0.5, "new_track_th_m": 0.1,
                     "marginal_timeout": (7 * fps // 10),
                     "central_timeout": fps,
                     "horizontal_margin": width // 10,
                     "vertical_margin": height // 10,
                     "frame_width": width,
                     "frame_height": height}
# Instantiate a tracker
tracker = SFSORT(tracker_arguments)
# Define a color list for track visualization
colors = {}
# Define the moving average window size (e.g., 5 frames)
window_size = 5
# Initialize a dictionary to store the moving average values for each track_id
moving_avg_dict = {}
# Create a dictionary to store the last update time for each track_id
last_update_times = {}

# Process each frame of the video
while cap.isOpened():
   ret, frame = cap.read()
   if not ret:
         break

   start_time = time.time()

   # Detect people in the frame
   prediction = session.predict(frame, imgsz=640, conf=0.1, iou=0.45,
                                half=False, max_det=99, verbose=False)

   # Exclude additional information from the predictions
   prediction_results = prediction[0].boxes.cpu().numpy()

   start_tracker_time = time.time()
   # Update the tracker with the latest detections
   tracks = tracker.update(prediction_results.xyxy, prediction_results.conf)
   end_tracker_time = time.time() - start_tracker_time

   # Skip additional analysis if the tracker is not currently tracking anyone
   if len(tracks) == 0:
      out.write(frame)
      continue
   
   # Extract tracking data from the tracker
   bbox_list = tracks[:, 0]
   track_id_list = tracks[:, 1]

   # Visualize tracks
   start_postprocess_time = time.time()
   for idx, (track_id, bbox) in enumerate(zip(track_id_list, bbox_list)):
      # Find the corresponding detection in the outputs array
      detection_idx = np.where(prediction_results.xyxy == bbox)[0][0]
      cls_id = int(prediction_results.cls[detection_idx])  # Get the current class_id value
      score = round(float(prediction_results.conf[detection_idx]), 2)

      # Define a new color for newly detected tracks
      if track_id not in colors:
         colors[track_id] = (random.randrange(255),
                             random.randrange(255),
                             random.randrange(255))

      color = colors[track_id]

      # Extract the bounding box coordinates
      x0, y0, x1, y1 = map(int, bbox)

      # Calculate the moving average of class_id values for this track_id
      if track_id in moving_avg_dict:
         moving_avg_dict[track_id].append(cls_id)
         last_update_times[track_id] = time.time()
         if len(moving_avg_dict[track_id]) > window_size:
            moving_avg_dict[track_id].pop(0)
         smoothed_cls_id = int(sum(moving_avg_dict[track_id]) / len(moving_avg_dict[track_id]))
      else:
         moving_avg_dict[track_id] = [cls_id]
         last_update_times[track_id] = time.time()
         smoothed_cls_id = cls_id

      name = names[smoothed_cls_id]
      name += ' '+str(score)

      # Draw the bounding boxes on the frame
      annotated_frame = cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)
      cv2.putText(annotated_frame, name+' '+str(track_id),
                  (x0, y0-5),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, thickness=2) 

   # After some time remove track_ids from moving_avg algo
   remove_stale_keys(moving_avg_dict,
                     last_update_times,
                     30)
   # Measure and visualize timers
   end_postprocess_time = time.time() - start_postprocess_time
   elapsed_time = time.time() - start_time
   fps = 1 / elapsed_time
   cv2.putText(annotated_frame, f'{fps:.1f} FPS (overall)',
      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA,)
   cv2.putText(annotated_frame, f'{end_tracker_time*1000:.2f} ms (SFSORT)',
      (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA,)
   cv2.putText(annotated_frame, f'{end_postprocess_time*1000:.2f} ms (post-process)',
      (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA,)

   # If key is pressed, close the window
   key = cv2.waitKey(1)
   if key == 27: # ESC
      break
   
   # cv2.imshow("test", annotated_frame)

   # Write the frame to the output video file
   out.write(annotated_frame)

# Release everything when done
cap.release()
out.release()
cv2.destroyAllWindows()
