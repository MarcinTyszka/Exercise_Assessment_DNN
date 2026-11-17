import os
import time
import tempfile
import urllib.request
import cv2
import torch
import numpy as np
import streamlit as st
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from feature_extraction import (
    extract_features_from_landmarks,
    create_or_load_dataset,
    EXERCISE_CLASSES,
    NUM_FEATURES
)
from model import PoseBaseMLP

MODEL_PATH = "pose_landmarker_lite.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"

POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (27, 31),
    (24, 26), (26, 28), (28, 30), (28, 32)
]


def get_safe_device():
    # Selects CUDA if supported by current PyTorch install, otherwise CPU
    if torch.cuda.is_available():
        try:
            torch.zeros(1, device="cuda")
            return torch.device("cuda")
        except Exception:
            pass
    return torch.device("cpu")


def resize_frame(rgb_image, target_width):
    # Resizes video frame proportionally to target_width
    h, w, _ = rgb_image.shape
    if w == target_width:
        return rgb_image
    scale = target_width / float(w)
    target_height = int(h * scale)
    return cv2.resize(rgb_image, (target_width, target_height), interpolation=cv2.INTER_AREA)


def draw_pose_skeleton(rgb_image, landmarks, is_correct=True):
    # Draws MediaPipe body joints and bone connections colored by form status
    h, w, _ = rgb_image.shape
    points = {}
    bone_color = (0, 230, 100) if is_correct else (255, 60, 60)

    for idx, lm in enumerate(landmarks):
        cx, cy = int(lm.x * w), int(lm.y * h)
        points[idx] = (cx, cy)
        if idx >= 11:
            cv2.circle(rgb_image, (cx, cy), 4, (255, 255, 0), -1)

    for start_idx, end_idx in POSE_CONNECTIONS:
        if start_idx in points and end_idx in points:
            cv2.line(rgb_image, points[start_idx], points[end_idx], bone_color, 2)


@st.cache_resource
def load_trained_model(checkpoint_path="checkpoints/latest_demo_model.pt"):
    # Loads trained model checkpoint along with feature standardization stats
    device = get_safe_device()
    feat_mean = np.zeros(NUM_FEATURES, dtype=np.float32)
    feat_std = np.ones(NUM_FEATURES, dtype=np.float32)

    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        hidden_dim = ckpt.get("hparams", {}).get("hidden_dim", 128)
        model = PoseBaseMLP(input_dim=NUM_FEATURES, hidden_dim=hidden_dim, num_exercises=len(EXERCISE_CLASSES))
        model.load_state_dict(ckpt["model_state_dict"])
        feat_mean = ckpt.get("feat_mean", feat_mean)
        feat_std = ckpt.get("feat_std", feat_std)
        status = f"Loaded trained checkpoint (Epoch {ckpt['epoch']}, Val Loss: {ckpt['loss']:.4f})"
    else:
        model = PoseBaseMLP(input_dim=NUM_FEATURES, hidden_dim=128, num_exercises=len(EXERCISE_CLASSES))
        status = "Warning: No checkpoint found. Run train.py first."

    model.to(device)
    model.eval()
    return model, device, feat_mean, feat_std, status


def evaluate_biomechanical_penalties(pred_ex, raw_feature_vec):
    # Evaluates biomechanical posture rules to penalize form score on clear postural faults
    bio = raw_feature_vec[-12:]
    l_knee_deg, r_knee_deg = bio[6] * 180.0, bio[7] * 180.0
    torso_lean_deg = bio[8] * 180.0
    knee_valgus_ratio = bio[9] * 2.5
    asymmetry = bio[11] * 180.0

    penalty = 0.0
    reasons = []

    if pred_ex in ["Squat", "Lunge"]:
        if torso_lean_deg > 42.0:
            penalty += 0.35
            reasons.append(f"Excessive forward torso lean ({torso_lean_deg:.0f} deg)")
        if knee_valgus_ratio < 0.72:
            penalty += 0.35
            reasons.append("Knee valgus (knees caving inward)")
        if asymmetry > 22.0:
            penalty += 0.25
            reasons.append(f"Left/Right knee-hip asymmetry ({asymmetry:.0f} deg)")
    elif pred_ex == "Push-up":
        l_hip_deg, r_hip_deg = bio[4] * 180.0, bio[5] * 180.0
        if min(l_hip_deg, r_hip_deg) < 135.0:
            penalty += 0.40
            reasons.append("Hip sag / pike posture error")
    elif asymmetry > 28.0:
        penalty += 0.30
        reasons.append(f"High joint asymmetry ({asymmetry:.0f} deg)")

    return penalty, reasons


def predict_single_feature_vector(model, device, feat_mean, feat_std, raw_feature_vec):
    # Standardizes input vector, runs neural inference, and applies biomechanical verification
    norm_vec = (raw_feature_vec - feat_mean) / feat_std
    tensor_in = torch.tensor(norm_vec, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        ex_logits, form_logit = model(tensor_in)
        ex_probs = torch.softmax(ex_logits, dim=1).cpu().numpy()[0]
        nn_form_prob = torch.sigmoid(form_logit).cpu().item()

    pred_idx = int(np.argmax(ex_probs))
    pred_ex = EXERCISE_CLASSES[pred_idx]

    penalty, reasons = evaluate_biomechanical_penalties(pred_ex, raw_feature_vec)
    final_form_prob = float(np.clip(nn_form_prob - penalty, 0.01, 0.99))

    return pred_ex, float(ex_probs[pred_idx]), final_form_prob, reasons


def process_uploaded_video(video_bytes, model, device, feat_mean, feat_std, threshold, process_width=640):
    # Processes uploaded video, overlays skeleton and form telemetry, and caches frames for replay
    if not os.path.exists(MODEL_PATH):
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(video_bytes)
    tfile.close()

    cap = cv2.VideoCapture(tfile.name)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    processed_frames = []
    error_frame_count = 0

    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5
    )

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame = resize_frame(rgb_frame, process_width)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            detection_result = landmarker.detect(mp_image)

            summary_text = "No person detected in frame."
            if detection_result.pose_landmarks and len(detection_result.pose_landmarks) > 0:
                landmarks = detection_result.pose_landmarks[0]
                features, angles_deg = extract_features_from_landmarks(landmarks)
                pred_ex, ex_conf, form_prob, reasons = predict_single_feature_vector(
                    model, device, feat_mean, feat_std, features
                )
                is_correct = form_prob >= threshold
                if not is_correct:
                    error_frame_count += 1

                form_label = "CORRECT FORM" if is_correct else "FORM ERROR"
                draw_pose_skeleton(rgb_frame, landmarks, is_correct=is_correct)

                cv2.putText(
                    rgb_frame,
                    f"{pred_ex} ({ex_conf * 100:.0f}%) | {form_label} ({form_prob * 100:.0f}%)",
                    (15, 32),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.68,
                    (0, 230, 80) if is_correct else (255, 50, 50),
                    2
                )

                reason_str = f" | **Issue:** {', '.join(reasons)}" if reasons else ""
                summary_text = (
                    f"**Predicted Exercise:** {pred_ex} ({ex_conf * 100:.1f}%) | "
                    f"**Form Evaluation:** {form_label} (Score: {form_prob * 100:.1f}%) | "
                    f"**Knee L/R:** {angles_deg[4]:.0f}° / {angles_deg[5]:.0f}° | "
                    f"**Torso Lean:** {angles_deg[6]:.0f}°{reason_str}"
                )

            bgr_annotated = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
            _, encoded_jpg = cv2.imencode(".jpg", bgr_annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            processed_frames.append((encoded_jpg.tobytes(), summary_text))

    cap.release()
    os.remove(tfile.name)
    return processed_frames, fps, error_frame_count


st.set_page_config(page_title="DNNDA Phase 2 - Exercise Assessment Demo", layout="wide")
st.title("Intelligent System for Exercise Assessment & Correction (Phase 2 Demo)")

model, device, feat_mean, feat_std, model_status = load_trained_model()
st.info(model_status)

mode = st.sidebar.radio(
    "Select Test Data Input Mode:",
    ["Pre-existing Test Dataset Sample", "Upload Workout Video (.mp4)"]
)
form_threshold = st.sidebar.slider(
    "Form Strictness Threshold", min_value=0.20, max_value=0.85, value=0.50, step=0.05
)

if mode == "Pre-existing Test Dataset Sample":
    st.subheader("Evaluate on Pre-existing Test Data")
    X, y_ex, y_form = create_or_load_dataset()

    sample_idx = st.slider("Select Test Sample Index", min_value=0, max_value=len(X) - 1, value=0)
    sample_features = X[sample_idx]
    true_exercise = EXERCISE_CLASSES[int(y_ex[sample_idx])]
    true_form = "Correct Form" if y_form[sample_idx] == 1.0 else "Form Error Detected"

    pred_exercise, ex_conf, form_prob, reasons = predict_single_feature_vector(
        model, device, feat_mean, feat_std, sample_features
    )
    pred_form = "Correct Form" if form_prob >= form_threshold else "Form Error Detected"

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Ground Truth Label**")
        st.write(f"Exercise: **{true_exercise}**")
        st.write(f"Form Status: **{true_form}**")

    with col2:
        st.markdown("**Base Model Prediction**")
        st.write(f"Predicted Exercise: **{pred_exercise}** (Confidence: {ex_conf * 100:.1f}%)")
        st.write(f"Predicted Form: **{pred_form}** (Score: {form_prob * 100:.1f}%)")
        if reasons:
            st.warning("Biomechanical flags: " + ", ".join(reasons))

    st.markdown("**Extracted Biomechanical Metrics**")
    bio = sample_features[-12:]
    metric_names = [
        "Left Elbow", "Right Elbow", "Left Hip", "Right Hip",
        "Left Knee", "Right Knee", "Torso Lean", "Asymmetry"
    ]
    metric_vals = [
        bio[0]*180, bio[1]*180, bio[4]*180, bio[5]*180,
        bio[6]*180, bio[7]*180, bio[8]*180, bio[11]*180
    ]
    st.table({name: [f"{val:.1f} deg"] for name, val in zip(metric_names, metric_vals)})

else:
    st.subheader("Upload Video for Real-Time MediaPipe + Base Model Inference")
    display_width = st.sidebar.slider("Video Display Width (px)", min_value=280, max_value=900, value=480, step=20)
    uploaded_file = st.file_uploader("Upload MP4 / AVI Exercise Video", type=["mp4", "avi", "mov"])

    if uploaded_file is not None:
        file_key = f"{uploaded_file.name}_{uploaded_file.size}_{form_threshold}"

        if st.session_state.get("video_key") != file_key:
            with st.spinner("Processing video frames with MediaPipe Pose..."):
                frames, fps, err_cnt = process_uploaded_video(
                    uploaded_file.getvalue(), model, device, feat_mean, feat_std, form_threshold, process_width=640
                )
                st.session_state["video_key"] = file_key
                st.session_state["processed_frames"] = frames
                st.session_state["video_fps"] = fps
                st.session_state["error_count"] = err_cnt
                st.session_state["should_play"] = True

        frames = st.session_state.get("processed_frames", [])
        fps = st.session_state.get("video_fps", 25.0)
        err_cnt = st.session_state.get("error_count", 0)

        if len(frames) > 0:
            err_ratio = (err_cnt / len(frames)) * 100.0
            if err_ratio >= 20.0:
                st.error(f"Overall Video Verdict: FORM ERROR DETECTED ({err_cnt}/{len(frames)} frames flagged, {err_ratio:.1f}%)")
            else:
                st.success(f"Overall Video Verdict: CORRECT FORM ({len(frames) - err_cnt}/{len(frames)} frames valid)")

            ctrl_col1, ctrl_col2 = st.columns([1, 4])
            with ctrl_col1:
                if st.button("Replay Video"):
                    st.session_state["should_play"] = True

            frame_placeholder = st.empty()
            metrics_placeholder = st.empty()

            if st.session_state.get("should_play", False):
                delay = 1.0 / min(fps, 30.0)
                for jpg_bytes, summary in frames:
                    frame_placeholder.image(jpg_bytes, width=display_width)
                    metrics_placeholder.markdown(summary)
                    time.sleep(delay)
                st.session_state["should_play"] = False

            with ctrl_col2:
                selected_idx = st.slider(
                    "Inspect Frame",
                    min_value=0,
                    max_value=len(frames) - 1,
                    value=len(frames) - 1
                )

            jpg_bytes, summary = frames[selected_idx]
            frame_placeholder.image(jpg_bytes, width=display_width)
            metrics_placeholder.markdown(summary)