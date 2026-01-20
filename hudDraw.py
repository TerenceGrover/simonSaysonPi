import cv2

def draw_hud(frame, score, streak, detected_groups, detected_compound, msg_top, msg_mid, msg_bot, color=(255,255,255)):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0,0), (w, 90), (0,0,0), -1)
    cv2.putText(frame, f"SCORE: {score}   STREAK: {streak}", (20, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255,255,255), 3)

    # detected status bar
    y = h - 90
    cv2.rectangle(frame, (0,y), (w, h), (0,0,0), -1)

    dg = " | ".join([f"{k}:{detected_groups.get(k,'?').replace('_',' ')}" for k in ["arms","legs","torso"]])
    cv2.putText(frame, dg[:120], (20, y+35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200,200,200), 2)

    comp = detected_compound if detected_compound else "None"
    cv2.putText(frame, f"compound: {comp}", (20, y+70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200,200,200), 2)

    # big messages
    cv2.putText(frame, msg_top, (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.4, color, 4)
    cv2.putText(frame, msg_mid, (20, 215), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3)
    cv2.putText(frame, msg_bot, (20, 270), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)