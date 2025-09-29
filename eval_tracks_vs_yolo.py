import numpy as np
from collections import defaultdict
import argparse
from ultralytics import YOLO
from SFSORT_adaptive import SFSORT
import cv2
import tqdm
import random
import ffmpeg


def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) between two bounding boxes."""
    x1, y1, x2, y2 = box1
    x1_, y1_, x2_, y2_ = box2
    
    xi1 = max(x1, x1_)
    yi1 = max(y1, y1_)
    xi2 = min(x2, x2_)
    yi2 = min(y2, y2_)
    
    inter_area = max(xi2 - xi1, 0) * max(yi2 - yi1, 0)
    
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_ - x1_) * (y2_ - y1_)
    
    union_area = box1_area + box2_area - inter_area
    
    iou = inter_area / union_area if union_area > 0 else 0
    return iou

def calculate_id_switches(tracks):
    """Calculate the number of ID switches."""
    id_switches = 0
    track_history = defaultdict(list)
    
    for frame, frame_tracks in tracks.items():
        for track in frame_tracks:
            track_id = track[1]  # track_id is at index 1 in SFSORT output
            bbox = track[0]
            x, y = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2  # Center of bounding box
            
            matched = False
            for prev_id, prev_pos in track_history.items():
                if np.linalg.norm(np.array([x, y]) - prev_pos[-1]) < 50:  # Arbitrary threshold
                    if prev_id != track_id:
                        id_switches += 1
                    track_history[track_id] = track_history.pop(prev_id)
                    matched = True
                    break
            
            if not matched:
                track_history[track_id] = []
            
            track_history[track_id].append(np.array([x, y]))
    
    return id_switches

def main(args):
    # Load YOLO model
    model = YOLO(args.model)
    
    # Load video
    cap = cv2.VideoCapture(args.video)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    process = (
        ffmpeg
        .input('pipe:', format='rawvideo', pix_fmt='bgr24', s='{}x{}'.format(width, height), framerate=fps)
        .output("output_sfsort.mp4", vcodec='libx264', pix_fmt='yuv420p', loglevel="quiet")
        .overwrite_output()
        .run_async(pipe_stdin=True)
    )

    # Initialize SFSORT tracker
    tracker_arguments = {
        "dynamic_tuning": True, "cth": 0.5,
        "high_th": 0.5, "high_th_m": 0.1,
        "match_th_first": 0.6, "match_th_first_m": 0.05,
        "match_th_second": 0.4, "low_th": 0.2,
        "new_track_th": 0.5, "new_track_th_m": 0.1,
        "marginal_timeout": (7 * fps // 10),
        "central_timeout": fps,
        "horizontal_margin": width // 10,
        "vertical_margin": height // 10,
        "frame_width": width,
        "frame_height": height,
    }
    tracker = SFSORT(tracker_arguments)

    yolo_detections = {}
    sfsort_tracks = {}
    total_yolo_detections = 0
    total_sfsort_tracks = 0

    # Define a color list for track visualization
    colors = {}

    for frame_num in tqdm.tqdm(range(length)):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run YOLO detection
        results = model(frame, imgsz=1280, conf=args.conf_threshold, iou=args.iou_threshold, max_det=100, verbose=False)
        
        # Filter detections based on specified classes
        detections = results[0].boxes.cpu().numpy()

        if args.classes:
            mask = np.isin(detections.cls, args.classes)
            detections = detections[mask]
            
        yolo_detections[frame_num] = detections
        total_yolo_detections += len(detections)
        print(total_yolo_detections)
        if total_yolo_detections > 1000:
            break
        
        # Run SFSORT tracker
        tracks = tracker.update(detections.xyxy, detections.conf, detections.cls)
        sfsort_tracks[frame_num] = tracks
        total_sfsort_tracks += len(tracks)

        # Skip additional analysis if the tracker is not currently tracking anyone
        if len(tracks) == 0:
            process.stdin.write(frame.tobytes())
            continue

        # Extract tracking data from the tracker
        bbox_list      = tracks[:, 0]
        track_id_list  = tracks[:, 1]
        cls_id_list    = tracks[:, 2]
        scores_list    = tracks[:, 3]

        # Visualize tracks
        for _, (track_id, bbox, cls_id, score) in enumerate(
            zip(track_id_list, bbox_list, cls_id_list, scores_list)):
            # Define a new color for newly detected tracks
            if track_id not in colors:
                colors[track_id] = (random.randrange(255),
                                    random.randrange(255),
                                    random.randrange(255))
                
            color = colors[track_id]
            # Extract the bounding box coordinates
            x0, y0, x1, y1 = map(int, bbox)
            score = str(score)
            # Draw the bounding boxes on the frame
            annotated_frame = cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)
            cv2.putText(annotated_frame, score+' '+str(track_id),
                        (x0, y0-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, thickness=2) 

        # If key is pressed, close the window
        key = cv2.waitKey(1)
        if key == 27: # ESC
            break

        # Write the frame to the output video file
        process.stdin.write(annotated_frame.tobytes())

    # Calculate metrics
    id_switches = calculate_id_switches(sfsort_tracks)
    
    print(f"Total YOLO detections: {total_yolo_detections}")
    print(f"Total SFSORT tracks: {total_sfsort_tracks}")
    print(f"Difference: {total_sfsort_tracks - total_yolo_detections}")
    print(f"ID Switches: {id_switches}")

    # Release everything when done
    cap.release()
    process.stdin.close()
    process.wait()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate SFSORT tracker against YOLO detections")
    parser.add_argument("--model", type=str, required=True, help="Path to YOLO model")
    parser.add_argument("--video", type=str, required=True, help="Path to input video")
    parser.add_argument("--conf-threshold", type=float, default=0.25, help="Confidence threshold for YOLO")
    parser.add_argument("--iou-threshold", type=float, default=0.45, help="IoU threshold for YOLO")
    parser.add_argument("--classes", nargs="+", type=int, help="Filter specific classes (e.g., 0 2 3)")
    
    args = parser.parse_args()
    main(args)
    