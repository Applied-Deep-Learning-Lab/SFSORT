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

def calculate_id_switches(tracks, tracker_type):
    """Calculate the number of ID switches."""
    id_switches = 0
    track_history = defaultdict(list)
    
    for frame, frame_tracks in tracks.items():
        for track in frame_tracks:
            if tracker_type == 'SFSORT':
                bbox, track_id = track[0], track[1]
            elif tracker_type == 'BoT-SORT':
                bbox = track.xyxy[0]
                print(bbox)
                track_id = int(track.id)  # Assuming track_id is at index 4 for BoT-SORT
            else:
                raise ValueError(f"Unknown tracker type: {tracker_type}")
            
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

    # Prepare pipes to ffmpeg for both SFSORT and BoT-SORT outputs
    process_sfsort = (
        ffmpeg
        .input('pipe:', format='rawvideo', pix_fmt='bgr24', s='{}x{}'.format(width, height), framerate=fps)
        .output("output_sfsort.mp4", vcodec='libx264', pix_fmt='yuv420p', loglevel="quiet")
        .overwrite_output()
        .run_async(pipe_stdin=True)
    )

    process_botsort = (
        ffmpeg
        .input('pipe:', format='rawvideo', pix_fmt='bgr24', s='{}x{}'.format(width, height), framerate=fps)
        .output("output_botsort.mp4", vcodec='libx264', pix_fmt='yuv420p', loglevel="quiet")
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
    sfsort_tracker = SFSORT(tracker_arguments)

    yolo_detections = {}
    sfsort_tracks = {}
    botsort_tracks = {}
    total_yolo_detections = 0
    total_sfsort_tracks = 0
    total_botsort_tracks = 0

    # Define color lists for track visualization
    colors_sfsort = {}
    colors_botsort = {}

    for frame_num in tqdm.tqdm(range(length)):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Create copies of the frame for SFSORT and BoT-SORT visualizations
        frame_sfsort = frame.copy()
        frame_botsort = frame.copy()
        
        # Run YOLO detection
        results = model(frame, imgsz=1280, conf=args.conf_threshold, iou=args.iou_threshold, max_det=100, verbose=False)
        
        # Filter detections based on specified classes
        detections = results[0].boxes.cpu().numpy()

        if args.classes:
            mask = np.isin(detections.cls, args.classes)
            detections = detections[mask]
            
        yolo_detections[frame_num] = detections
        total_yolo_detections += len(detections)
        
        # Run SFSORT tracker
        sfsort_tracks_frame = sfsort_tracker.update(detections.xyxy, detections.conf, detections.cls)
        sfsort_tracks[frame_num] = sfsort_tracks_frame
        total_sfsort_tracks += len(sfsort_tracks_frame)

        # Run BoT-SORT tracker
        botsort_results = model.track(frame, persist=True, verbose=False)
        botsort_tracks_frame = botsort_results[0].cpu().numpy().boxes
        botsort_tracks[frame_num] = botsort_tracks_frame
        total_botsort_tracks += len(botsort_tracks_frame)

        # Visualize SFSORT tracks
        for _, (track_id, bbox, cls_id, score) in enumerate(zip(sfsort_tracks_frame[:, 1], sfsort_tracks_frame[:, 0], sfsort_tracks_frame[:, 2], sfsort_tracks_frame[:, 3])):
            if track_id not in colors_sfsort:
                colors_sfsort[track_id] = (random.randrange(255), random.randrange(255), random.randrange(255))
            
            color = colors_sfsort[track_id]
            x0, y0, x1, y1 = map(int, bbox)
            cv2.rectangle(frame_sfsort, (x0, y0), (x1, y1), color, 2)
            cv2.putText(frame_sfsort, f"{score:.2f} {track_id}", (x0, y0-5), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, thickness=2)

        # Visualize BoT-SORT tracks
        for track in botsort_tracks_frame:
            track_id = int(track.id)
            if track_id not in colors_botsort:
                colors_botsort[track_id] = (random.randrange(255), random.randrange(255), random.randrange(255))
            
            color = colors_botsort[track_id]
            x0, y0, x1, y1 = map(int, track.xyxy[0])
            cv2.rectangle(frame_botsort, (x0, y0), (x1, y1), color, 2)
            cv2.putText(frame_botsort, f"{track.conf[0]:.2f} {track_id}", (x0, y0-5), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, thickness=2)

        # Write the frames to the output videos
        process_sfsort.stdin.write(frame_sfsort.tobytes())
        process_botsort.stdin.write(frame_botsort.tobytes())

    # Calculate metrics
    sfsort_id_switches = calculate_id_switches(sfsort_tracks, 'SFSORT')
    botsort_id_switches = calculate_id_switches(botsort_tracks, 'BoT-SORT')
    
    print(f"Total YOLO detections: {total_yolo_detections}")
    print(f"Total SFSORT tracks: {total_sfsort_tracks}")
    print(f"Total BoT-SORT tracks: {total_botsort_tracks}")
    print(f"SFSORT difference from YOLO: {total_sfsort_tracks - total_yolo_detections}")
    print(f"BoT-SORT difference from YOLO: {total_botsort_tracks - total_yolo_detections}")
    print(f"SFSORT ID Switches: {sfsort_id_switches}")
    print(f"BoT-SORT ID Switches: {botsort_id_switches}")

    # Release everything when done
    cap.release()
    process_sfsort.stdin.close()
    process_sfsort.wait()
    process_botsort.stdin.close()
    process_botsort.wait()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate SFSORT and BoT-SORT trackers against YOLO detections")
    parser.add_argument("--model", type=str, required=True, help="Path to YOLO model")
    parser.add_argument("--video", type=str, required=True, help="Path to input video")
    parser.add_argument("--conf-threshold", type=float, default=0.25, help="Confidence threshold for YOLO")
    parser.add_argument("--iou-threshold", type=float, default=0.45, help="IoU threshold for YOLO")
    parser.add_argument("--classes", nargs="+", type=int, help="Filter specific classes (e.g., 0 2 3)")
    
    args = parser.parse_args()
    main(args)