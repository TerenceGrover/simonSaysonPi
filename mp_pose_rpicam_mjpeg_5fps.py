#!/usr/bin/env python3
import os
import time
import subprocess
import cv2
import numpy as np
import mediapipe as mp

# --------------------
# CONFIG
# --------------------
W, H = 854, 480
FPS = 5
FRAME_TIME = 1.0 / FPS
WINDOW_NAME = "MediaPipe Pose"

# If you run from SSH and want window on HDMI:
os.environ.setdefault("DISPLAY", ":0")

# rpicam-vid MJPEG stream to stdout
RPICAM_CMD = [
    "rpicam-vid",
    "--timeout", "0",
    "--nopreview",
    "--width", str(W),
    "--height", str(H),
    "--framerate", str(FPS),
    "--codec", "mjpeg",
    "--inline",  # helps some decoders; harmless otherwise
    "-o", "-"    # stdout
]

# --------------------
# MEDIAPIPE
# --------------------
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=0,
    smooth_landmarks=True,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, 1280, 720)
cv2.moveWindow(WINDOW_NAME, 0, 0)
def mjpeg_frames_from_pipe(pipe):
    """
    Yield JPEG byte blobs from an MJPEG stream (stdout).
    Looks for JPEG SOI/EOI markers: 0xFFD8 ... 0xFFD9
    """
    buf = bytearray()
    while True:
        chunk = pipe.read(4096)
        if not chunk:
            break
        buf.extend(chunk)

        # Find JPEG start/end
        while True:
            start = buf.find(b"\xff\xd8")
            if start == -1:
                # Keep buffer from growing forever
                if len(buf) > 1_000_000:
                    del buf[:-100_000]
                break
            end = buf.find(b"\xff\xd9", start + 2)
            if end == -1:
                # Need more data
                if start > 0:
                    del buf[:start]
                break

            jpg = bytes(buf[start:end + 2])
            del buf[:end + 2]
            yield jpg

def main():
    print("🎥 Starting rpicam-vid MJPEG stream...")
    proc = subprocess.Popen(
        RPICAM_CMD,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        bufsize=0
    )

    try:
        print("✅ MediaPipe Pose running. Press 'q' to quit.")
        for jpg in mjpeg_frames_from_pipe(proc.stdout):
            t0 = time.time()

            # Decode JPEG -> BGR frame
            arr = np.frombuffer(jpg, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)

            if frame is None:
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            if results.pose_landmarks:
                mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            cv2.putText(frame, "5 FPS CAP", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

            cv2.imshow(WINDOW_NAME, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            # Hard FPS cap (extra safety)
            elapsed = time.time() - t0
            sleep_for = FRAME_TIME - elapsed
            if sleep_for > 0:
                time.sleep(sleep_for)

    finally:
        print("👋 Shutting down...")
        try:
            proc.terminate()
        except Exception:
            pass
        try:
            proc.wait(timeout=2)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

        pose.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
