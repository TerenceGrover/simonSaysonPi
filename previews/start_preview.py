#!/usr/bin/env python3
import os
import time
import cv2
from picamera2 import Picamera2

# MediaPipe
import mediapipe as mp

def main():
    os.environ.setdefault("DISPLAY", ":0")

    TARGET_FPS = 5.0
    FRAME_TIME = 1.0 / TARGET_FPS

    # Lower res = less heat. You can bump if it’s too ugly.
    WIDTH, HEIGHT = 640, 480

    cam = Picamera2()
    config = cam.create_preview_configuration(main={"size": (WIDTH, HEIGHT), "format": "RGB888"})
    cam.configure(config)
    cam.start()

    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles

    # model_complexity=0 is lighter (good for Pi)
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=0,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cv2.namedWindow("Pose (5 FPS cap)", cv2.WINDOW_NORMAL)

    print("✅ MediaPipe Pose running at hard-capped 5 FPS. Press 'q' to quit.")
    try:
        while True:
            t0 = time.time()

            frame = cam.capture_array()          # RGB
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # MediaPipe expects RGB
            results = pose.process(frame)

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame_bgr,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style()
                )

            # Tiny HUD so you can see it’s obeying you
            cv2.putText(frame_bgr, "5 FPS CAP", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

            cv2.imshow("Pose (5 FPS cap)", frame_bgr)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

            # Hard FPS cap
            elapsed = time.time() - t0
            sleep_for = FRAME_TIME - elapsed
            if sleep_for > 0:
                time.sleep(sleep_for)

    except KeyboardInterrupt:
        pass
    finally:
        pose.close()
        cam.stop()
        cv2.destroyAllWindows()
        print("👋 Clean exit.")

if __name__ == "__main__":
    main()
