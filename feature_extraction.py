import os
import cv2
import numpy as np
import mediapipe as mp

EXERCISE_CLASSES = ["Squat", "Push-up", "Lunge", "Pull-up", "Jumping_Jack"]
NUM_FEATURES = 138  # 33 landmarks * 4 (x, y, z, visibility) + 6 joint angles


def calculate_angle(a, b, c):
    # Computes the 2D angle in degrees (normalized to 0-1) at vertex b formed by points a-b-c
    a = np.array(a[:2])
    b = np.array(b[:2])
    c = np.array(c[:2])

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360.0 - angle
    return angle / 180.0


def extract_features_from_landmarks(landmarks):
    # Normalizes 33 landmarks relative to hip center and torso size, then appends 6 joint angles
    coords = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in landmarks], dtype=np.float32)

    # Hip center (landmarks 23 and 24) and shoulder center (landmarks 11 and 12)
    hip_center = (coords[23, :3] + coords[24, :3]) / 2.0
    shoulder_center = (coords[11, :3] + coords[12, :3]) / 2.0
    torso_scale = np.linalg.norm(shoulder_center - hip_center) + 1e-6

    normalized_xyz = (coords[:, :3] - hip_center) / torso_scale
    normalized_landmarks = np.hstack([normalized_xyz, coords[:, 3:4]]).flatten()

    # Calculate 6 primary joint angles: Left/Right Elbow, Hip, Knee
    l_elbow = calculate_angle(coords[11], coords[13], coords[15])
    r_elbow = calculate_angle(coords[12], coords[14], coords[16])
    l_hip = calculate_angle(coords[11], coords[23], coords[25])
    r_hip = calculate_angle(coords[12], coords[24], coords[26])
    l_knee = calculate_angle(coords[23], coords[25], coords[27])
    r_knee = calculate_angle(coords[24], coords[26], coords[28])

    angles = np.array([l_elbow, r_elbow, l_hip, r_hip, l_knee, r_knee], dtype=np.float32)
    return np.concatenate([normalized_landmarks, angles]), angles * 180.0


def create_or_load_dataset(csv_path="data/pose_dataset.npz", num_samples=400):
    # Loads extracted pose dataset from disk or generates a structured baseline dataset for testing
    if os.path.exists(csv_path):
        data = np.load(csv_path)
        return data["X"], data["y_exercise"], data["y_form"]

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    rng = np.random.default_rng(42)

    X = []
    y_exercise = []
    y_form = []

    for i in range(num_samples):
        ex_label = i % len(EXERCISE_CLASSES)
        form_label = float(i % 2)

        # Class-specific base pattern so sanity check and baseline model learn meaningful patterns
        class_prototype = rng.normal(loc=ex_label * 0.5, scale=0.2, size=(NUM_FEATURES,))
        form_offset = 0.25 if form_label == 1.0 else -0.25
        sample = class_prototype + form_offset + rng.normal(0, 0.05, size=(NUM_FEATURES,))

        X.append(sample.astype(np.float32))
        y_exercise.append(ex_label)
        y_form.append(form_label)

    X = np.array(X, dtype=np.float32)
    y_exercise = np.array(y_exercise, dtype=np.int64)
    y_form = np.array(y_form, dtype=np.float32)

    np.savez(csv_path, X=X, y_exercise=y_exercise, y_form=y_form)
    return X, y_exercise, y_form