import os
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


def draw_pose_skeleton(rgb_image, landmarks):
    # Draws body keypoints and bone connections directly using OpenCV
    h, w, _ = rgb_image.shape
    points = {}
    for idx, lm in enumerate(landmarks):
        cx, cy = int(lm.x * w), int(lm.y * h)
        points[idx] = (cx, cy)
        if idx >= 11:
            cv2.circle(rgb_image, (cx, cy), 4, (0, 255, 255), -1)

    for start_idx, end_idx in POSE_CONNECTIONS:
        if start_idx in points and end_idx in points:
            cv2.line(rgb_image, points[start_idx], points[end_idx], (255, 100, 0), 2)


@st.cache_resource
def load_trained_model(checkpoint_path="checkpoints/latest_demo_model.pt"):
    # Loads the trained PyTorch checkpoint or initializes weights if no checkpoint exists yet
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        hidden_dim = ckpt.get("hparams", {}).get("hidden_dim", 128)
        model = PoseBaseMLP(input_dim=NUM_FEATURES, hidden_dim=hidden_dim, num_exercises=len(EXERCISE_CLASSES))
        model.load_state_dict(ckpt["model_state_dict"])
        status = f"Loaded trained checkpoint (Epoch {ckpt['epoch']}, Loss: {ckpt['loss']:.4f})"
    else:
        model = PoseBaseMLP(input_dim=NUM_FEATURES, hidden_dim=128, num_exercises=len(EXERCISE_CLASSES))
        status = "Warning: No checkpoint found. Run train.py first."

    model.to(device)
    model.eval()
    return model, device, status


def predict_single_feature_vector(model, device, feature_vec):
    # Runs forward inference on a single 138-dim pose vector and returns probabilities
    tensor_in = torch.tensor(feature_vec, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        ex_logits, form_logit = model(tensor_in)
        ex_probs = torch.softmax(ex_logits, dim=1).cpu().numpy()[0]
        form_prob = torch.sigmoid(form_logit).cpu().item()

    pred_idx = int(np.argmax(ex_probs))
    return EXERCISE_CLASSES[pred_idx], float(ex_probs[pred_idx]), form_prob


st.set_page_config(page_title="DNNDA Phase 2 - Exercise Assessment Demo", layout="wide")
st.title("Intelligent System for Exercise Assessment & Correction (Phase 2 Demo)")

model, device, model_status = load_trained_model()
st.info(model_status)

mode = st.sidebar.radio(
    "Select Test Data Input Mode:",
    ["Pre-existing Test Dataset Sample", "Upload Workout Video (.mp4)"]
)

if mode == "Pre-existing Test Dataset Sample":
    st.subheader("Evaluate on Pre-existing Test Data")
    X, y_ex, y_form = create_or_load_dataset()

    sample_idx = st.slider("Select Test Sample Index", min_value=0, max_value=len(X) - 1, value=0)
    sample_features = X[sample_idx]
    true_exercise = EXERCISE_CLASSES[int(y_ex[sample_idx])]
    true_form = "Correct Form" if y_form[sample_idx] == 1.0 else "Form Error Detected"

    pred_exercise, ex_conf, form_prob = predict_single_feature_vector(model, device, sample_features)
    pred_form = "Correct Form" if form_prob >= 0.5 else "Form Error Detected"

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Ground Truth Label**")
        st.write(f"Exercise: **{true_exercise}**")
        st.write(f"Form Status: **{true_form}**")

    with col2:
        st.markdown("**Base Model Prediction**")
        st.write(f"Predicted Exercise: **{pred_exercise}** (Confidence: {ex_conf * 100:.1f}%)")
        st.write(f"Predicted Form: **{pred_form}** (Score: {form_prob * 100:.1f}%)")

    st.markdown("**Extracted Joint Angles (Degrees)**")
    angles_deg = sample_features[-6:] * 180.0
    angle_names = ["Left Elbow", "Right Elbow", "Left Hip", "Right Hip", "Left Knee", "Right Knee"]
    st.table({name: [f"{val:.1f} deg"] for name, val in zip(angle_names, angles_deg)})

else:
    st.subheader("Upload Video for Real-Time MediaPipe + Base Model Inference")
    uploaded_file = st.file_uploader("Upload MP4 / AVI Exercise Video", type=["mp4", "avi", "mov"])

    if uploaded_file is not None:
        if not os.path.exists(MODEL_PATH):
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())
        tfile.close()

        cap = cv2.VideoCapture(tfile.name)
        frame_placeholder = st.empty()
        metrics_placeholder = st.empty()

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
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                detection_result = landmarker.detect(mp_image)

                if detection_result.pose_landmarks and len(detection_result.pose_landmarks) > 0:
                    landmarks = detection_result.pose_landmarks[0]
                    draw_pose_skeleton(rgb_frame, landmarks)
                    features, angles_deg = extract_features_from_landmarks(landmarks)
                    pred_ex, ex_conf, form_prob = predict_single_feature_vector(model, device, features)
                    form_label = "CORRECT FORM" if form_prob >= 0.5 else "FORM ERROR"

                    cv2.putText(
                        rgb_frame,
                        f"{pred_ex} ({ex_conf*100:.0f}%) | {form_label}",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 255, 0) if form_prob >= 0.5 else (255, 50, 50),
                        2
                    )
                    metrics_placeholder.markdown(
                        f"**Predicted Exercise:** {pred_ex} ({ex_conf*100:.1f}%) | "
                        f"**Form Evaluation:** {form_label} ({form_prob*100:.1f}%) | "
                        f"**Left Knee Angle:** {angles_deg[4]:.1f} deg | "
                        f"**Right Knee Angle:** {angles_deg[5]:.1f} deg"
                    )

                frame_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)

        cap.release()