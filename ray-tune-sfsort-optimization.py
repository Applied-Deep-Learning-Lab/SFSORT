import cv2
import numpy as np
import onnxruntime as ort
from SFSORT import SFSORT
from yolov7_onnxruntime_sfsort import letterbox
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler

# Global variables
MODEL_PATH = "C:/Users/virus/SFSORT/cartype_v2.onnx"
VIDEO_PATH = "C:/Users/virus/SFSORT/test.mp4"

# All classes
names = ['bike', 'bus', 'car', 'construction equipment', 'emergency', 'motorbike', 'personal mobility', 'quad bike', 'truck']

def evaluate_tracker(config):
    # Model loading
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    session = ort.InferenceSession(MODEL_PATH, providers=providers)
    # Load the video file
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Organize tracker arguments
    tracker_arguments = {
        "dynamic_tuning": True,
        "cth": config["cth"],
        "high_th": config["high_th"],
        "high_th_m": config["high_th_m"],
        "match_th_first": config["match_th_first"],
        "match_th_first_m": config["match_th_first_m"],
        "match_th_second": config["match_th_second"],
        "low_th": config["low_th"],
        "new_track_th": config["new_track_th"],
        "new_track_th_m": config["new_track_th_m"],
        "marginal_timeout": int(config["marginal_timeout"] * fps),
        "central_timeout": int(config["central_timeout"] * fps),
        "horizontal_margin": int(width * config["horizontal_margin_ratio"]),
        "vertical_margin": int(height * config["vertical_margin_ratio"]),
        "frame_width": 640,
        "frame_height": 640
    }

    # Instantiate a tracker
    tracker = SFSORT(tracker_arguments)

    total_detections = 0
    total_tracks = 0
    track_id_changes = 0
    previous_track_ids = set()

    # Process each frame of the video
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocessing steps
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = img.copy()
        image, _, _ = letterbox(image, auto=False)
        image = image.transpose((2, 0, 1))
        image = np.expand_dims(image, 0)
        image = np.ascontiguousarray(image)
        im = image.astype(np.float32)
        im /= 255

        # ONNX inference
        outname = [i.name for i in session.get_outputs()]
        inname = [i.name for i in session.get_inputs()]
        inp = {inname[0]: im}
        outputs = session.run(outname, inp)[0]

        # Update the tracker with the latest detections
        tracks = tracker.update(outputs[:, 1:5], outputs[:, 6], outputs[:, 5])

        total_detections += len(outputs)
        total_tracks += len(tracks)

        # Count track ID changes
        current_track_ids = set(track[1] for track in tracks)
        track_id_changes += len(current_track_ids.symmetric_difference(previous_track_ids))
        previous_track_ids = current_track_ids

    cap.release()

    # Calculate the score
    detection_ratio = total_tracks / total_detections if total_detections > 0 else 0
    track_consistency = 1 / (1 + track_id_changes)  # Inverse of track ID changes, normalized
    score = (detection_ratio + track_consistency) / 2  # Average of both metrics

    # Report the score to Ray Tune
    return {"score": score}

def trial_str_creator(trial):
    return f"{trial.trial_id}_sfsort"

def main():
    ray.init()

    # Define the parameter search space
    config = {
        "cth": tune.uniform(0.3, 0.7),
        "high_th": tune.uniform(0.6, 0.9),
        "high_th_m": tune.uniform(0.05, 0.2),
        "match_th_first": tune.uniform(0.3, 0.7),
        "match_th_first_m": tune.uniform(0.01, 0.1),
        "match_th_second": tune.uniform(0.05, 0.3),
        "low_th": tune.uniform(0.1, 0.3),
        "new_track_th": tune.uniform(0.2, 0.5),
        "new_track_th_m": tune.uniform(0.05, 0.2),
        "marginal_timeout": tune.uniform(0.5, 2.0),  # In seconds
        "central_timeout": tune.uniform(0.8, 3.0),  # In seconds
        "horizontal_margin_ratio": tune.uniform(0.05, 0.2),
        "vertical_margin_ratio": tune.uniform(0.05, 0.2),
    }

    # Set up the Ray Tune experiment
    tuner = tune.Tuner(
        evaluate_tracker,
        config=config,
        num_samples=500,  # Number of different configurations to try
        scheduler=ASHAScheduler(
            metric="score",
            mode="max",
            max_t=100,
            grace_period=1,
            reduction_factor=2
        ),
        resources_per_trial={"cpu": 4, "gpu": 1},
        trial_dirname_creator=trial_str_creator
    )

    # Run the experiment
    results = tuner.fit()

    # Get the best configuration
    best_result = results.get_best_result(metric="score", mode="max")
    best_config = best_result.config
    print("Best configuration:", best_config)
    print("Best score:", best_result.metrics["score"])

    # You can save the best configuration to a file
    with open("best_sfsort_config.txt", "w") as f:
        for key, value in best_config.items():
            f.write(f"{key}: {value}\n")

if __name__ == "__main__":
    main()
