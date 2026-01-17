#!/usr/bin/env python3
import os
import time
import cv2
import mediapipe as mp

# --------------------
# CONFIG
# --------------------
TARGET_FPS = 5
FRAME_TIME = 1.0 / TARGET_FPS
W, H = 640, 480

# Make the window appear on the HDMI desktop (from SSH)
os.environ.setdefault("DISPLAY", ":0")

# libcamera -> NV12 (supported) -> convert -> BGR -> appsink for OpenCV
PIPELINE = (
    f"libcamerasrc ! "
    f"video/x-raw,format=NV12,width={W},height={H},framerate={TARGET_FPS}/1 ! "
    f"videoconvert ! "
    f"video/x-raw,format=BGR ! "
    f"appsink drop=true max-buffers=1 sync=false"
)

# Force GStreamer backend
cap = cv2.VideoCapture(PIPELINE, cv2.CAP_GSTREAMER)
if not cap.isOpened():
    raise RuntimeError("Could not open camera via GStreamer pipeline. "
                       "Check gstreamer1.0-libcamera install and pipeline caps.")

# --------------------
# MEDIAPIPE POSE
# --------------------
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=0,        # lightest model = less heat
    smooth_landmarks=True,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cv2.namedWindow("MediaPipe Pose (NV12, 5 FPS)", cv2.WINDOW_NORMAL)

print("✅ MediaPipe Pose running (libcamera→GStreamer NV12→OpenCV). Press 'q' to quit.")

try:
    while True:
        t0 = time.time()

        ret, frame = cap.read()
        if not ret or frame is None:
            print("⚠️ Frame grab failed (GStreamer/appsink)")
            time.sleep(0.2)
            continue

        # MediaPipe wants RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        if results.pose_landmarks:
            mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        cv2.putText(
            frame,
            "5 FPS CAP",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2
        )

        cv2.imshow("MediaPipe Pose (NV12, 5 FPS)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        # Hard FPS cap (extra safety beyond pipeline framerate)
        elapsed = time.time() - t0
        sleep_for = FRAME_TIME - elapsed
        if sleep_for > 0:
            time.sleep(sleep_for)

finally:
    cap.release()
    pose.close()
    cv2.destroyAllWindows()
    print("👋 Clean exit")
