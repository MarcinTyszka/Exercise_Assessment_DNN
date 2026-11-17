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
    EXERCISE_CLASSES,
    NUM_BIOMECH_FEATURES
)

RAW_VIDEOS_DIR = "data/raw_videos"
OUTPUT_NPZ_PATH = "data/pose_dataset.npz"
MODEL_PATH = "pose_landmarker_lite.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"


def ensure_pose_model():
    # Downloads the MediaPipe PoseLandmarker model file if missing
    if not os.path.exists(MODEL_PATH):
        print(f"Downloading {MODEL_PATH}...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)


def create_folder_structure():
    # Creates directory structure for all exercises and form labels
    for exercise in EXERCISE_CLASSES:
        os.makedirs(os.path.join(RAW_VIDEOS_DIR, exercise, "correct"), exist_ok=True)
        os.makedirs(os.path.join(RAW_VIDEOS_DIR, exercise, "error"), exist_ok=True)


def process_video_file(video_path, landmarker, frame_skip=2):
    # Extracts 144-dim pose and biomechanical vectors from video frames
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


def create_biomechanical_flawed_copy(X_corr, rng):
    # Generates realistic flawed exercise frames (shallow depth, rounded back, knee valgus, asymmetry)
    n = len(X_corr)
    X_bad = X_corr.copy()

    for i in range(n):
        mistake_type = i % 3
        bio = X_bad[i, -NUM_BIOMECH_FEATURES:].copy()

        if mistake_type == 0:
            # Shallow rep / incomplete range of motion + excessive forward torso lean
            bio[0:8] = np.clip(bio[0:8] + rng.uniform(0.18, 0.32, size=8), 0.0, 1.0)
            bio[8] = np.clip(bio[8] + rng.uniform(0.22, 0.40), 0.0, 1.0)
            X_bad[i, 11*4 : 13*4 : 4] += rng.uniform(0.15, 0.30)
            X_bad[i, 11*4 + 1 : 13*4 + 1 : 4] += rng.uniform(0.15, 0.30)
        elif mistake_type == 1:
            # Knee valgus (knees caving inward) + left-right asymmetry
            bio[9] = np.clip(bio[9] * rng.uniform(0.45, 0.65), 0.0, 1.0)
            bio[11] = np.clip(bio[11] + rng.uniform(0.20, 0.35), 0.0, 1.0)
            X_bad[i, 25*4] *= 0.55
            X_bad[i, 26*4] *= 0.55
        else:
            # Hip sag / poor posture alignment across torso and lower body
            bio[4:6] = np.clip(bio[4:6] - rng.uniform(0.18, 0.30, size=2), 0.0, 1.0)
            bio[8] = np.clip(bio[8] + rng.uniform(0.20, 0.35), 0.0, 1.0)
            bio[11] = np.clip(bio[11] + rng.uniform(0.15, 0.30), 0.0, 1.0)

        X_bad[i, -NUM_BIOMECH_FEATURES:] = bio

    return X_bad


def balance_dataset_per_exercise(X, y_exercise, y_form):
    # Ensures every exercise class has an equal number of correct and error frames
    rng = np.random.default_rng(42)
    X_all, y_ex_all, y_form_all = [X], [y_exercise], [y_form]

    for ex_idx in range(len(EXERCISE_CLASSES)):
        ex_mask = (y_exercise == ex_idx)
        corr_idx = np.where(ex_mask & (y_form == 1.0))[0]
        err_idx = np.where(ex_mask & (y_form == 0.0))[0]

        n_corr = len(corr_idx)
        n_err = len(err_idx)

        if n_corr > n_err and n_corr > 0:
            deficit = n_corr - n_err
            source_indices = rng.choice(corr_idx, size=deficit, replace=True)
            X_syn_err = create_biomechanical_flawed_copy(X[source_indices], rng)

            X_all.append(X_syn_err)
            y_ex_all.append(np.full(deficit, ex_idx, dtype=np.int64))
            y_form_all.append(np.zeros(deficit, dtype=np.float32))

    return (
        np.vstack(X_all).astype(np.float32),
        np.concatenate(y_ex_all).astype(np.int64),
        np.concatenate(y_form_all).astype(np.float32)
    )


def build_dataset_from_videos():
    # Builds balanced pose dataset from raw_videos and saves to data/pose_dataset.npz
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

                    if len(vectors) > 6:
                        # Trim first and last 10% of frames to reduce idle standing noise
                        trim = max(1, int(len(vectors) * 0.10))
                        vectors = vectors[trim:-trim]

                    for vec in vectors:
                        X.append(vec)
                        y_exercise.append(ex_idx)
                        y_form.append(form_label)

    if len(X) == 0:
        print("No videos found in data/raw_videos/. Generating synthetic fallback dataset.")
        create_or_load_dataset(OUTPUT_NPZ_PATH)
        return

    X = np.array(X, dtype=np.float32)
    y_exercise = np.array(y_exercise, dtype=np.int64)
    y_form = np.array(y_form, dtype=np.float32)

    X, y_exercise, y_form = balance_dataset_per_exercise(X, y_exercise, y_form)

    os.makedirs(os.path.dirname(OUTPUT_NPZ_PATH), exist_ok=True)
    np.savez(OUTPUT_NPZ_PATH, X=X, y_exercise=y_exercise, y_form=y_form)

    print("-" * 60)
    print(f"Saved {len(X)} total frames to {OUTPUT_NPZ_PATH}")
    print(f"Class balance -> Correct (1.0): {int(np.sum(y_form == 1.0))} | Error (0.0): {int(np.sum(y_form == 0.0))}")
    for idx, name in enumerate(EXERCISE_CLASSES):
        c_cnt = int(np.sum((y_exercise == idx) & (y_form == 1.0)))
        e_cnt = int(np.sum((y_exercise == idx) & (y_form == 0.0)))
        print(f"  - {name:14s}: {c_cnt:4d} correct | {e_cnt:4d} error")
    print("-" * 60)


if __name__ == "__main__":
    build_dataset_from_videos()