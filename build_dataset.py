import os
import urllib.request
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from feature_extraction import (
    extract_features_from_landmarks,
    create_or_load_dataset,
    EXERCISE_CLASSES
)

RAW_VIDEOS_DIR = "data/raw_videos"
OUTPUT_NPZ_PATH = "data/pose_dataset.npz"
MODEL_PATH = "pose_landmarker_lite.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"


def ensure_pose_model():
    # Downloads the official MediaPipe PoseLandmarker model file if not present locally
    if not os.path.exists(MODEL_PATH):
        print(f"Downloading {MODEL_PATH}...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)


def create_folder_structure():
    # Creates the directory tree for all 5 exercises and both form labels (correct/error)
    for exercise in EXERCISE_CLASSES:
        os.makedirs(os.path.join(RAW_VIDEOS_DIR, exercise, "correct"), exist_ok=True)
        os.makedirs(os.path.join(RAW_VIDEOS_DIR, exercise, "error"), exist_ok=True)


def process_video_file(video_path, landmarker, frame_skip=2):
    # Reads a video file and extracts normalized 138-dim pose feature vectors from frames
    cap = cv2.VideoCapture(video_path)
    extracted_vectors = []
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_skip == 0:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            detection_result = landmarker.detect(mp_image)

            if detection_result.pose_landmarks and len(detection_result.pose_landmarks) > 0:
                landmarks = detection_result.pose_landmarks[0]
                features, _ = extract_features_from_landmarks(landmarks)
                extracted_vectors.append(features)

        frame_idx += 1

    cap.release()
    return extracted_vectors


def build_dataset_from_videos():
    # Iterates through all exercise folders, extracts pose features, and saves to pose_dataset.npz
    ensure_pose_model()
    create_folder_structure()

    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5
    )

    X, y_exercise, y_form = [], [], []

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        for ex_idx, exercise_name in enumerate(EXERCISE_CLASSES):
            for form_name, form_label in [("correct", 1.0), ("error", 0.0)]:
                folder_path = os.path.join(RAW_VIDEOS_DIR, exercise_name, form_name)
                video_files = [
                    f for f in os.listdir(folder_path)
                    if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))
                ]

                for vf in video_files:
                    full_path = os.path.join(folder_path, vf)
                    print(f"Processing: {full_path}")
                    vectors = process_video_file(full_path, landmarker, frame_skip=2)

                    for vec in vectors:
                        X.append(vec)
                        y_exercise.append(ex_idx)
                        y_form.append(form_label)

    if len(X) == 0:
        print("No videos found in data/raw_videos/. Generating synthetic fallback dataset instead.")
        create_or_load_dataset(OUTPUT_NPZ_PATH)
        return

    X = np.array(X, dtype=np.float32)
    y_exercise = np.array(y_exercise, dtype=np.int64)
    y_form = np.array(y_form, dtype=np.float32)

    os.makedirs(os.path.dirname(OUTPUT_NPZ_PATH), exist_ok=True)
    np.savez(OUTPUT_NPZ_PATH, X=X, y_exercise=y_exercise, y_form=y_form)
    print(f"Saved {len(X)} labeled frames to {OUTPUT_NPZ_PATH}")


if __name__ == "__main__":
    build_dataset_from_videos()