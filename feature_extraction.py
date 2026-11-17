import os
import numpy as np

EXERCISE_CLASSES = ["Squat", "Push-up", "Lunge", "Pull-up", "Jumping_Jack"]
NUM_BIOMECH_FEATURES = 12
NUM_FEATURES = 33 * 4 + NUM_BIOMECH_FEATURES  # 144 total features


def calculate_angle(a, b, c):
    # Computes the interior 2D angle at vertex b normalized to [0, 1]
    a = np.array(a[:2], dtype=np.float32)
    b = np.array(b[:2], dtype=np.float32)
    c = np.array(c[:2], dtype=np.float32)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360.0 - angle
    return float(angle / 180.0)


def extract_features_from_landmarks(landmarks):
    # Normalizes 33 landmarks and computes 12 biomechanical posture and angle features
    coords = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in landmarks], dtype=np.float32)

    hip_center = (coords[23, :3] + coords[24, :3]) / 2.0
    shoulder_center = (coords[11, :3] + coords[12, :3]) / 2.0
    torso_vec = shoulder_center - hip_center
    torso_scale = np.linalg.norm(torso_vec) + 1e-6

    normalized_xyz = np.clip((coords[:, :3] - hip_center) / torso_scale, -3.0, 3.0)
    normalized_landmarks = np.hstack([normalized_xyz, coords[:, 3:4]]).flatten()

    # 8 Primary Joint Angles (normalized 0 to 1)
    l_elbow = calculate_angle(coords[11], coords[13], coords[15])
    r_elbow = calculate_angle(coords[12], coords[14], coords[16])
    l_shoulder = calculate_angle(coords[23], coords[11], coords[13])
    r_shoulder = calculate_angle(coords[24], coords[12], coords[14])
    l_hip = calculate_angle(coords[11], coords[23], coords[25])
    r_hip = calculate_angle(coords[12], coords[24], coords[26])
    l_knee = calculate_angle(coords[23], coords[25], coords[27])
    r_knee = calculate_angle(coords[24], coords[26], coords[28])

    # 4 Posture & Alignment Features (Torso Lean, Knee Valgus Ratio, Wrist-Ankle Span, Asymmetry)
    vertical_ref = hip_center[:2] + np.array([0.0, -1.0], dtype=np.float32)
    torso_lean = calculate_angle(vertical_ref, hip_center[:2], shoulder_center[:2])

    knee_width = np.linalg.norm(coords[25, :2] - coords[26, :2])
    ankle_width = np.linalg.norm(coords[27, :2] - coords[28, :2]) + 1e-4
    knee_alignment_ratio = float(np.clip(knee_width / ankle_width, 0.0, 2.5) / 2.5)

    wrist_center = (coords[15, :2] + coords[16, :2]) / 2.0
    ankle_center = (coords[27, :2] + coords[28, :2]) / 2.0
    body_extension = float(np.clip(np.linalg.norm(wrist_center - ankle_center) / (torso_scale * 3.0), 0.0, 1.5))

    symmetry_error = float(
        (abs(l_knee - r_knee) + abs(l_hip - r_hip) + abs(l_elbow - r_elbow)) / 3.0
    )

    biomech = np.array([
        l_elbow, r_elbow, l_shoulder, r_shoulder,
        l_hip, r_hip, l_knee, r_knee,
        torso_lean, knee_alignment_ratio, body_extension, symmetry_error
    ], dtype=np.float32)

    display_angles_deg = np.array([
        l_elbow * 180.0, r_elbow * 180.0,
        l_hip * 180.0, r_hip * 180.0,
        l_knee * 180.0, r_knee * 180.0,
        torso_lean * 180.0, knee_alignment_ratio * 100.0
    ], dtype=np.float32)

    return np.concatenate([normalized_landmarks, biomech]), display_angles_deg


def create_or_load_dataset(npz_path="data/pose_dataset.npz", num_samples=500):
    # Loads the extracted dataset or creates a balanced fallback dataset
    if os.path.exists(npz_path):
        data = np.load(npz_path)
        if data["X"].shape[1] == NUM_FEATURES:
            return data["X"], data["y_exercise"], data["y_form"]

    os.makedirs(os.path.dirname(npz_path), exist_ok=True)
    rng = np.random.default_rng(42)

    X, y_exercise, y_form = [], [], []
    for i in range(num_samples):
        ex_label = i % len(EXERCISE_CLASSES)
        form_label = float(i % 2)

        base = rng.normal(loc=ex_label * 0.35, scale=0.15, size=(NUM_FEATURES,)).astype(np.float32)
        if form_label == 1.0:
            base[-NUM_BIOMECH_FEATURES:] = np.clip(rng.normal(0.55, 0.08, size=(NUM_BIOMECH_FEATURES,)), 0.0, 1.0)
        else:
            base[-NUM_BIOMECH_FEATURES:] = np.clip(rng.normal(0.82, 0.10, size=(NUM_BIOMECH_FEATURES,)), 0.0, 1.0)

        X.append(base)
        y_exercise.append(ex_label)
        y_form.append(form_label)

    X = np.array(X, dtype=np.float32)
    y_exercise = np.array(y_exercise, dtype=np.int64)
    y_form = np.array(y_form, dtype=np.float32)

    np.savez(npz_path, X=X, y_exercise=y_exercise, y_form=y_form)
    return X, y_exercise, y_form