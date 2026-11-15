import os
import time
import argparse
import cv2
from feature_extraction import EXERCISE_CLASSES


def record_clip(exercise_idx, form_type, duration_sec=5, camera_id=0):
    # Records a short webcam video clip and saves it directly into the labeled folder
    exercise_name = EXERCISE_CLASSES[exercise_idx]
    save_dir = os.path.join("data", "raw_videos", exercise_name, form_type)
    os.makedirs(save_dir, exist_ok=True)

    filename = f"clip_{int(time.time())}.mp4"
    output_path = os.path.join(save_dir, filename)

    cap = cv2.VideoCapture(camera_id)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
    fps = 20.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"Get ready! Recording {exercise_name} ({form_type}) in 3 seconds...")
    time.sleep(3)
    print("Recording started...")

    start_time = time.time()
    while (time.time() - start_time) < duration_sec:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        cv2.imshow("Recording (Press Q to stop early)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Saved clip to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record short exercise clips via webcam")
    parser.add_argument("--exercise", type=int, default=0, help="0:Squat, 1:Push-up, 2:Lunge, 3:Pull-up, 4:Jumping Jack")
    parser.add_argument("--form", type=str, default="correct", choices=["correct", "error"])
    parser.add_argument("--duration", type=int, default=5, help="Clip duration in seconds")
    args = parser.parse_args()

    record_clip(args.exercise, args.form, args.duration)