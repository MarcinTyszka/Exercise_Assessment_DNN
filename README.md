# Intelligent System for Exercise Assessment & Correction

Real-time physical exercise classification and posture/form evaluation using **MediaPipe Pose**, **PyTorch**, and **Streamlit**.

## Authors
* **Marcin Tyszka**
* **Maciej Pajkowski**

## Project Overview
The system extracts 33 3D body landmarks via MediaPipe Pose (`PoseLandmarker`), normalizes coordinates relative to hip center and torso scale, and computes 8 primary joint angles plus 4 posture/alignment metrics (**144 features per frame**). A two-head Base Multi-Layer Perceptron (`PoseBaseMLP` without regularization) simultaneously predicts:
1. **Exercise Type** (5 classes: *Squat, Push-up, Lunge, Pull-up, Jumping Jack*) using `CrossEntropyLoss`.
2. **Form Correctness** (Binary flag: *Correct Form* vs. *Form Error*) using class-weighted `BCEWithLogitsLoss`.

## Repository Structure
* `feature_extraction.py` – MediaPipe landmark normalization, 8 joint angles, and 4 posture/symmetry feature calculations.
* `record_webcam.py` – CLI helper to capture labeled `.mp4` webcam clips directly into `data/raw_videos/`.
* `build_dataset.py` – Automated video processing pipeline converting raw `.mp4` clips into `data/pose_dataset.npz` with video-level IDs.
* `model.py` – PyTorch `PoseBaseMLP` architecture and multi-task loss function (`PoseAssessmentLoss`).
* `train.py` – Video-grouped 70/15/15 stratified training loop with GPU/CPU support, single-batch overfitting sanity check (`--sanity_check`), held-out test evaluation (Accuracy & F1-score), model checkpointing, searchable `.txt` logs, and TensorBoard + HParams logging.
* `app.py` – Interactive Streamlit web application supporting held-out test sample inspection and `.mp4` video uploads.
* `logs/` – Plain-text training and sanity check logs.
* `runs/` – TensorBoard event files and HParams dashboard logs.
* `checkpoints/` – Saved PyTorch model weights (`.pt`).

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

*(Note: For NVIDIA RTX 50-series / Blackwell GPUs, install PyTorch with CUDA 12.8 support, or run on CPU via `$env:CUDA_VISIBLE_DEVICES="-1"`).*

## Usage

### 1. Build Dataset from Videos (Optional if `data/pose_dataset.npz` is already present)
Place `.mp4` videos inside `data/raw_videos/<Exercise>/<correct|error>/` and run:

```bash
python build_dataset.py
```

### 2. Run Single-Batch Overfitting Sanity Check
Verify that the model overfits a single batch to 100% accuracy and logs hyperparameters:

```bash
python train.py --sanity_check --epochs 100 --lr 0.005 --batch_size 16 --optimizer Adam
python train.py --sanity_check --epochs 100 --lr 0.01 --batch_size 16 --optimizer SGD
```

### 3. Run Full Baseline Training
Train the base model on the full dataset to generate `checkpoints/latest_demo_model.pt` for the Streamlit app:

```bash
python train.py --epochs 80 --lr 0.003 --batch_size 16 --optimizer Adam
```

### 4. Inspect TensorBoard & HParams Dashboard
Launch TensorBoard and open `http://localhost:6006` in your browser:

```bash
tensorboard --logdir runs
```

### 5. Launch Streamlit Demo Application
Start the interactive web application:

```bash
streamlit run app.py
```

In the Streamlit sidebar, you can switch between:
* **Pre-existing Test Dataset Sample** – inspect predictions and extracted joint angles on saved dataset frames.
* **Upload Workout Video (.mp4)** – upload an exercise video to view real-time MediaPipe skeleton tracking, joint angles, and neural network predictions.