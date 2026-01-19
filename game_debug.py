#!/usr/bin/env python3
import os
import time
import json
import random
import subprocess
from pathlib import Path
from collections import defaultdict, deque

import cv2
import numpy as np
import mediapipe as mp

# ============================================================
# SIMON SAYS (Pose Edition)
# - Uses your pose JSONs (poses/) + compounds (compounds/)
# - rpicam-vid MJPEG pipe (fast + reliable on Pi)
# - Validator logic: visibility gating + constraints + scoring + smoothing
# ============================================================

# --------------------
# DISPLAY / CAMERA
# --------------------
os.environ.setdefault("DISPLAY", ":0")

W, H = 1280, 720
FPS = 10
FRAME_TIME = 1.0 / FPS
WINDOW_NAME = "SIMON SAYS (POSE)"
ROTATE = cv2.ROTATE_90_COUNTERCLOCKWISE  # or CLOCKWISE if wrong direction
AUDIO_ROOT = Path("audio")
SIMON_DIR = AUDIO_ROOT / "simon_says"
COMMANDS_DIR = AUDIO_ROOT / "commands"

RPICAM_CMD = [
    "rpicam-vid",
    "--timeout", "0",
    "--nopreview",
    "--width", str(W),
    "--height", str(H),
    "--framerate", str(FPS),
    "--codec", "mjpeg",
    "--inline",
    "-o", "-"
]

# --------------------
# GAME TUNING
# --------------------
SIMON_SAYS_PROB = 0.72     # probability the prompt is a real "Simon says"
PROMPT_SEC      = 0.2      # show prompt before evaluation starts
TIME_LIMIT_SEC  = 2.5      # time allowed to achieve the pose
HOLD_SEC        = 0.5      # must hold correct pose this long (stable frames)
COOLDOWN_SEC    = 0.5      # pause between rounds
SPEEDUP_EVERY   = 2       # every N successes, tighten timing
SPEEDUP_FACTOR  = 0.90     # multiply TIME_LIMIT/HOLD by this (gentle)

# --------------------
# POSE VALIDATOR TUNING (your defaults)
# --------------------
MAX_ACCEPTABLE_SCORE = {"arms": 2.8, "legs": 3.2, "torso": 3.3}
MARGIN_MIN          = {"arms": 0.45, "legs": 0.55, "torso": 0.60}

SMOOTH_WIN  = 1
SMOOTH_NEED = 1

VIS_CORE  = 0.50
VIS_WRIST = 0.18
VIS_ANKLE = 0.1
VIS_FACE  = 0.18

# --------------------
# ASSETS
# --------------------
POSE_DIR = Path("poses")
COMPOUNDS_DIR = Path("compounds")

# --------------------
# LANDMARK INDICES
# --------------------
LM = {
    "NOSE": 0,
    "L_EAR": 7,
    "R_EAR": 8,
    "L_SHO": 11,
    "R_SHO": 12,
    "L_ELB": 13,
    "R_ELB": 14,
    "L_WRI": 15,
    "R_WRI": 16,
    "L_HIP": 23,
    "R_HIP": 24,
    "L_KNE": 25,
    "R_KNE": 26,
    "L_ANK": 27,
    "R_ANK": 28,
}

def play_wav(path):
    if not path or not os.path.exists(path):
        return
    subprocess.run(
        ["aplay", str(path)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

def play_prompt_audio(target_name, simon):

    cmd_dir = COMMANDS_DIR / target_name
    if not cmd_dir.exists():
        print(f"[WARN] Missing command audio for {target_name}")
        return

    # Simon says (random variant)
    if simon:
        play_random_wav_from_folder(SIMON_DIR)
        play_random_wav_from_folder(cmd_dir)
    else:
        play_random_wav_from_folder(cmd_dir)

def thr_for_idx(idx: int) -> float:
    if idx in (LM["L_SHO"], LM["R_SHO"], LM["L_HIP"], LM["R_HIP"], LM["L_KNE"], LM["R_KNE"]):
        return VIS_CORE
    if idx in (LM["L_WRI"], LM["R_WRI"]):
        return VIS_WRIST
    if idx in (LM["L_ANK"], LM["R_ANK"]):
        return VIS_ANKLE
    if idx in (LM["NOSE"], LM["L_EAR"], LM["R_EAR"]):
        return VIS_FACE
    return 0.30

# ============================================================
# MJPEG PIPE DECODER
# ============================================================
def mjpeg_frames_from_pipe(pipe):
    buf = bytearray()
    while True:
        chunk = pipe.read(4096)
        if not chunk:
            break
        buf.extend(chunk)

        while True:
            start = buf.find(b"\xff\xd8")
            if start == -1:
                if len(buf) > 1_000_000:
                    del buf[:-100_000]
                break
            end = buf.find(b"\xff\xd9", start + 2)
            if end == -1:
                if start > 0:
                    del buf[:start]
                break

            jpg = bytes(buf[start:end + 2])
            del buf[:end + 2]
            yield jpg

# ============================================================
# LOADING POSES + COMPOUNDS
# ============================================================
POSES_BY_GROUP = {"arms": [], "legs": [], "torso": []}
COMPOUNDS = {}

def load_json(p: Path):
    with open(p, "r") as f:
        return json.load(f)

def sanitize_constraints(pose_def):
    cons = pose_def.get("constraints", {}) or {}
    feats = pose_def.get("features", {}) or {}
    safe = {}

    for k, rule in cons.items():
        if k not in feats:
            continue
        mean = feats[k].get("mean", None)
        tol  = feats[k].get("tol",  None)
        if mean is None or tol is None:
            continue

        rr = dict(rule)
        if "min" in rr and rr["min"] > (mean + tol * 2.5):
            rr.pop("min", None)
        if "max" in rr and rr["max"] < (mean - tol * 2.5):
            rr.pop("max", None)

        if rr:
            safe[k] = rr

    pose_def["_constraints_sanitized"] = safe

def load_assets():
    if POSE_DIR.exists():
        for f in POSE_DIR.iterdir():
            if f.suffix != ".json":
                continue
            data = load_json(f)
            g = data.get("group")
            if g not in POSES_BY_GROUP:
                continue
            data.setdefault("active_features", list(data.get("features", {}).keys()))
            data.setdefault("constraints", {})
            sanitize_constraints(data)
            POSES_BY_GROUP[g].append(data)

    if COMPOUNDS_DIR.exists():
        for f in COMPOUNDS_DIR.iterdir():
            if f.suffix != ".json":
                continue
            c = load_json(f)
            COMPOUNDS[c["name"]] = c["requires"]

    print("Loaded poses:", {g: len(POSES_BY_GROUP[g]) for g in POSES_BY_GROUP})
    print("Loaded compounds:", list(COMPOUNDS.keys()))

# ============================================================
# FEATURE EXTRACTION (your body-axes v4)
# ============================================================
def _angle(a, b, c) -> float:
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosang = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return float(np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0))))

def _dist(a, b) -> float:
    return float(np.linalg.norm(np.array(a) - np.array(b)))

def _unit(v):
    n = np.linalg.norm(v) + 1e-6
    return v / n

def _make_body_axes(hipC, shoC):
    up = _unit(shoC - hipC)
    if up[1] > 0:
        up = -up
    right = np.array([up[1], -up[0]], dtype=np.float32)
    right = _unit(right)
    return right, up

def _proj(pt, origin, right, up):
    v = pt - origin
    return float(np.dot(v, right)), float(np.dot(v, up))

def extract_features_and_vismap(lm):
    def p(i):
        return np.array([lm[i].x, lm[i].y], dtype=np.float32)

    def vis(i):
        return float(getattr(lm[i], "visibility", 1.0))

    vis_map = {i: vis(i) for i in range(len(lm))}

    LS, RS = p(LM["L_SHO"]), p(LM["R_SHO"])
    LE, RE = p(LM["L_ELB"]), p(LM["R_ELB"])
    LW, RW = p(LM["L_WRI"]), p(LM["R_WRI"])
    LH, RH = p(LM["L_HIP"]), p(LM["R_HIP"])
    LK, RK = p(LM["L_KNE"]), p(LM["R_KNE"])
    LA, RA = p(LM["L_ANK"]), p(LM["R_ANK"])
    NO  = p(LM["NOSE"])
    LEA, REA = p(LM["L_EAR"]), p(LM["R_EAR"])

    hipC_raw = (LH + RH) / 2.0
    shoulder_w = _dist(LS, RS) + 1e-6

    def n(pt):
        return (pt - hipC_raw) / shoulder_w

    LS0, RS0, LE0, RE0, LW0, RW0 = map(n, [LS, RS, LE, RE, LW, RW])
    LH0, RH0, LK0, RK0, LA0, RA0 = map(n, [LH, RH, LK, RK, LA, RA])
    NO0, LEA0, REA0 = map(n, [NO, LEA, REA])

    hipC = (LH0 + RH0) / 2.0
    shoC = (LS0 + RS0) / 2.0
    headC = (LEA0 + REA0) / 2.0

    right, up = _make_body_axes(hipC, shoC)

    def B(pt):
        x, y = _proj(pt, hipC, right, up)
        return np.array([x, y], dtype=np.float32)

    LSb, RSb, LEb, REb, LWb, RWb = map(B, [LS0, RS0, LE0, RE0, LW0, RW0])
    LHb, RHb, LKb, RKb, LAb, RAb = map(B, [LH0, RH0, LK0, RK0, LA0, RA0])
    NOb, HEADb = map(B, [NO0, headC])

    f = {}
    f["ang_elb_L"] = _angle(LSb, LEb, LWb)
    f["ang_elb_R"] = _angle(RSb, REb, RWb)
    f["ang_kne_L"] = _angle(LHb, LKb, LAb)
    f["ang_kne_R"] = _angle(RHb, RKb, RAb)
    f["ang_hip_L"] = _angle((LSb + RSb) / 2.0, LHb, LKb)
    f["ang_hip_R"] = _angle((LSb + RSb) / 2.0, RHb, RKb)

    f["h_wri_vs_sho_L"] = float(LWb[1] - LSb[1])
    f["h_wri_vs_sho_R"] = float(RWb[1] - RSb[1])
    f["h_wri_vs_head_L"] = float(LWb[1] - HEADb[1])
    f["h_wri_vs_head_R"] = float(RWb[1] - HEADb[1])
    f["h_ank_vs_hip_L"] = float(LAb[1] - LHb[1])
    f["h_ank_vs_hip_R"] = float(RAb[1] - RHb[1])

    f["d_wri_same_sho_L"] = _dist(LWb, LSb)
    f["d_wri_same_sho_R"] = _dist(RWb, RSb)
    f["d_wri_opp_sho_L"]  = _dist(LWb, RSb)
    f["d_wri_opp_sho_R"]  = _dist(RWb, LSb)

    f["d_wri_same_elb_L"] = _dist(LWb, LEb)
    f["d_wri_same_elb_R"] = _dist(RWb, REb)
    f["d_wri_opp_elb_L"]  = _dist(LWb, REb)
    f["d_wri_opp_elb_R"]  = _dist(RWb, LEb)

    f["d_wri_wri"] = _dist(LWb, RWb)
    f["d_ank_ank"] = _dist(LAb, RAb)

    f["d_wri_head_L"] = _dist(LWb, HEADb)
    f["d_wri_head_R"] = _dist(RWb, HEADb)
    f["d_wri_nose_L"] = _dist(LWb, NOb)
    f["d_wri_nose_R"] = _dist(RWb, NOb)

    f["d_wri_ank_L"] = _dist(LWb, LAb)
    f["d_wri_ank_R"] = _dist(RWb, RAb)
    f["d_wri_ank_min"] = float(min(f["d_wri_ank_L"], f["d_wri_ank_R"]))

    f["r_opp_same_sho_L"] = float(f["d_wri_opp_sho_L"] / (f["d_wri_same_sho_L"] + 1e-6))
    f["r_opp_same_sho_R"] = float(f["d_wri_opp_sho_R"] / (f["d_wri_same_sho_R"] + 1e-6))
    f["r_opp_same_elb_L"] = float(f["d_wri_opp_elb_L"] / (f["d_wri_same_elb_L"] + 1e-6))
    f["r_opp_same_elb_R"] = float(f["d_wri_opp_elb_R"] / (f["d_wri_same_elb_R"] + 1e-6))

    return f, vis_map

# ============================================================
# VISIBILITY + CONSTRAINTS + SCORING
# ============================================================
def required_landmarks_for_pose(pose_def):
    active = set(pose_def.get("active_features", []))
    req = {LM["L_HIP"], LM["R_HIP"], LM["L_SHO"], LM["R_SHO"]}

    if any(k.startswith("ang_elb") for k in active) or any("elb" in k for k in active):
        req |= {LM["L_ELB"], LM["R_ELB"]}
    if any(k.startswith("ang_kne") for k in active) or any("kne" in k for k in active):
        req |= {LM["L_KNE"], LM["R_KNE"]}
    if any("wri" in k for k in active) or any(k.startswith("ang_elb") for k in active):
        req |= {LM["L_WRI"], LM["R_WRI"]}
    if any("ank" in k for k in active):
        req |= {LM["L_ANK"], LM["R_ANK"]}
    if any("nose" in k for k in active):
        req |= {LM["NOSE"]}
    if any("head" in k for k in active):
        req |= {LM["L_EAR"], LM["R_EAR"]}

    return sorted(req)

def pose_visibility_ok(vis_map, pose_def):
    for idx in required_landmarks_for_pose(pose_def):
        if vis_map.get(idx, 0.0) < thr_for_idx(idx):
            return False
    return True

def passes_constraints(feats, pose_def):
    cons = pose_def.get("_constraints_sanitized", pose_def.get("constraints", {})) or {}
    for k, rule in cons.items():
        if k not in feats:
            return False
        v = feats[k]
        if "min" in rule and v < rule["min"]:
            return False
        if "max" in rule and v > rule["max"]:
            return False

    raw = pose_def.get("raw_name", "")
    if raw in ("touch_nose", "hand_on_head"):
        kL, kR = ("d_wri_nose_L", "d_wri_nose_R") if raw == "touch_nose" else ("d_wri_head_L", "d_wri_head_R")
        refL = pose_def["features"].get(kL, {"mean": 0.0, "tol": 0.6})
        refR = pose_def["features"].get(kR, {"mean": 0.0, "tol": 0.6})
        thr = min(refL["mean"] + refL["tol"] * 1.2, refR["mean"] + refR["tol"] * 1.2)
        if min(feats.get(kL, 9e9), feats.get(kR, 9e9)) > thr:
            return False

    return True

def score_pose(feats, pose_def):
    used = pose_def.get("active_features", [])
    if not used:
        return 1e9
    s, n = 0.0, 0
    for k in used:
        if k not in feats:
            continue
        ref = pose_def["features"].get(k)
        if not ref:
            continue
        s += abs(feats[k] - ref["mean"]) / (ref["tol"] + 1e-6)
        n += 1
    return (s / n) if n else 1e9

def detect_group(feats, vis_map, group_name):
    best_name = "UNKNOWN"
    best_score = 1e9
    second_score = 1e9
    candidates = 0
    vis_ok_ct = 0
    cons_ok_ct = 0

    for p in POSES_BY_GROUP[group_name]:
        candidates += 1

        if not pose_visibility_ok(vis_map, p):
            continue
        vis_ok_ct += 1

        if not passes_constraints(feats, p):
            continue
        cons_ok_ct += 1

        sc = score_pose(feats, p)

        # Guard: NaNs kill comparisons silently
        if not np.isfinite(sc):
            continue

        if sc < best_score:
            second_score = best_score
            best_score = sc
            best_name = p["pose_name"]
        elif sc < second_score:
            second_score = sc

    # no viable pose
    if best_score >= 1e8:
        return "UNKNOWN", best_score, second_score, (candidates, vis_ok_ct, cons_ok_ct), "NO_CANDIDATE"

    if best_score > MAX_ACCEPTABLE_SCORE[group_name]:
        return "UNKNOWN", best_score, second_score, (candidates, vis_ok_ct, cons_ok_ct), "SCORE_TOO_HIGH"

    if (second_score - best_score) < MARGIN_MIN[group_name]:
        return "UNKNOWN", best_score, second_score, (candidates, vis_ok_ct, cons_ok_ct), "MARGIN_TOO_SMALL"

    return best_name, best_score, second_score, (candidates, vis_ok_ct, cons_ok_ct), "OK"


# ============================================================
# COMPOUND MATCHING
# ============================================================
def matches_compound(detected, compound_name):
    if not compound_name:
        return True
    req = COMPOUNDS.get(compound_name)
    if not req:
        return False
    for g in ["arms", "legs", "torso"]:
        want = req.get(g, "ANY")
        if want == "ANY":
            continue
        if not want.startswith(g + "_"):
            want = f"{g}_{want}"
        if detected.get(g) != want:
            return False
    return True

# ============================================================
# POSE DETECTOR WRAPPER (adds smoothing + compound label)
# ============================================================
class PoseDetector:
    def __init__(self):
        self.hist = {g: deque(maxlen=SMOOTH_WIN) for g in ["arms", "legs", "torso"]}

    def _smooth(self, group, current):
        self.hist[group].append(current)
        if len(self.hist[group]) < SMOOTH_WIN:
            return current
        counts = defaultdict(int)
        for x in self.hist[group]:
            counts[x] += 1
        best, cnt = max(counts.items(), key=lambda kv: kv[1])
        return best if cnt >= SMOOTH_NEED else "UNKNOWN"

    def detect(self, pose_landmarks):
        detected = {"arms": "UNKNOWN", "legs": "UNKNOWN", "torso": "UNKNOWN"}
        dbg = {}

        feats, vis_map = extract_features_and_vismap(pose_landmarks.landmark)

        for g in ["arms", "legs", "torso"]:
            name, best, second, counts, reason = detect_group(feats, vis_map, g)
            name = self._smooth(g, name)
            detected[g] = name

            dbg[g] = {
                "best": best,
                "second": second,
                "counts": counts,
                "reason": reason
            }

        # Decide compound if any matches
        compound = None
        for cname in COMPOUNDS.keys():
            if matches_compound(detected, cname):
                compound = cname
                break

        return detected, compound, dbg, feats, vis_map


# ============================================================
# GAME: build a list of playable prompts
# ============================================================
def build_prompt_pool():
    """
    We’ll build prompts from:
      - atomic presets: any single group pose (arms_* / legs_* / torso_*)
      - compound presets: each compound by name
    """
    atomic = []
    for g in ["arms", "legs", "torso"]:
        for p in POSES_BY_GROUP[g]:
            atomic.append(("atomic", p["pose_name"]))

    compounds = [("compound", name) for name in COMPOUNDS.keys()]
    pool = atomic + compounds

    # If you want to exclude specific spicy poses, do it here:
    # pool = [x for x in pool if x[1] not in ("legs_split",)]
    random.shuffle(pool)
    return pool

def pretty_label(kind, name):
    if kind == "compound":
        return name.replace("_", " ").upper()
    # atomic: strip group prefix for UI but keep info
    return name.replace("_", " ").upper()

def current_matches_target(detected_groups, detected_compound, target_kind, target_name):
    if target_kind == "compound":
        return detected_compound == target_name
    # atomic: need the specific group to match
    # pose_name format is "arms_xxx" etc
    if "_" not in target_name:
        return False
    group = target_name.split("_", 1)[0]
    return detected_groups.get(group) == target_name

# ============================================================
# UI DRAW
# ============================================================
def draw_hud(frame, score, streak, detected_groups, detected_compound,
             msg_top, msg_mid, msg_bot, dbg=None, color=(255,255,255)):
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
    if dbg and "arms" in dbg:
        a = dbg["arms"]
        cands, vOK, cOK = a["counts"]
        best = a["best"]
        second = a["second"]
        reason = a["reason"]

        txt = (
            f"ARMS dbg | poses={cands} visOK={vOK} consOK={cOK} "
            f"best={best:.2f} second={second:.2f} {reason}"
        )

        cv2.putText(
            frame,
            txt[:120],
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

def read_frame(jpeg_stream, max_skip=5):
    frame = None
    for _ in range(max_skip):
        jpg = next(jpeg_stream, None)
        if jpg is None:
            return None
        arr = np.frombuffer(jpg, dtype=np.uint8)
        tmp = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if tmp is not None:
            frame = tmp
    if frame is None:
        return None
    return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

def play_random_wav_from_folder(folder: Path):
    if not folder.exists() or not folder.is_dir():
        print(f"[WARN] Missing audio folder: {folder}")
        return

    files = sorted(folder.glob("*.wav"))
    if not files:
        print(f"[WARN] No wav files in: {folder}")
        return

    wav = random.choice(files)
    play_wav(str(wav))

# ============================================================
# MAIN
# ============================================================
def main():
    load_assets()
    pool = build_prompt_pool()
    if not pool:
        raise RuntimeError("No poses found. Put JSONs in ./poses and (optionally) ./compounds")

    mp_pose = mp.solutions.pose
    mp_draw = mp.solutions.drawing_utils

    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    detector = PoseDetector()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1280, 720)
    cv2.moveWindow(WINDOW_NAME, 0, 0)

    print("🎥 Starting rpicam-vid MJPEG stream...")
    proc = subprocess.Popen(
        RPICAM_CMD,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        bufsize=0
    )
    jpeg_stream = mjpeg_frames_from_pipe(proc.stdout)

    score = 0
    streak = 0

    # difficulty params that change
    time_limit = TIME_LIMIT_SEC
    hold_sec   = HOLD_SEC

    time.sleep(2.0)  # let camera settle
    try:
        while True:
            # pick target
            target_kind, target_name = random.choice(pool)

            simon = (random.random() < SIMON_SAYS_PROB)
            play_prompt_audio(target_name, simon)
            prompt_text = ("SIMON SAYS: " if simon else "DO: ")
            label = pretty_label(target_kind, target_name)

            # prompt phase
            t_prompt_end = time.time() + PROMPT_SEC
            detected_groups = {"arms":"UNKNOWN","legs":"UNKNOWN","torso":"UNKNOWN"}
            detected_comp = None

            while time.time() < t_prompt_end:
                t0 = time.time()
                frame = read_frame(jpeg_stream)
                if frame is None:
                    continue

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = pose.process(rgb)

                if res.pose_landmarks:
                    mp_draw.draw_landmarks(frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                    detected_groups, detected_comp, dbg, feats, vis_map = detector.detect(res.pose_landmarks)

                draw_hud(
                    frame, score, streak, detected_groups, detected_comp,
                    msg_top=prompt_text + label,
                    msg_mid="Get ready...",
                    msg_bot=f"limit={time_limit:.1f}s hold={hold_sec:.1f}s",
                    dbg=dbg,
                    color=(255,255,255)
                )
                cv2.imshow(WINDOW_NAME, frame)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    return

                elapsed = time.time() - t0
                sleep_for = FRAME_TIME - elapsed
                if sleep_for > 0:
                    time.sleep(sleep_for)

            # evaluation phase
            t_deadline = time.time() + time_limit
            held_for = 0.0
            last_t = time.time()

            # success condition differs:
            # - simon says: must match + hold
            # - not simon: must NOT match (and we just wait out the timer)
            fail_reason = None
            success = False

            while time.time() < t_deadline:
                t0 = time.time()
                frame = read_frame(jpeg_stream)
                if frame is None:
                    continue

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = pose.process(rgb)

                if res.pose_landmarks:
                    mp_draw.draw_landmarks(frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                    detected_groups, detected_comp = detector.detect(res.pose_landmarks)
                else:
                    detected_groups = {"arms":"UNKNOWN","legs":"UNKNOWN","torso":"UNKNOWN"}
                    detected_comp = None

                now = time.time()
                dt = now - last_t
                last_t = now

                is_match = current_matches_target(detected_groups, detected_comp, target_kind, target_name)

                if simon:
                    if is_match:
                        held_for += dt
                    else:
                        held_for = max(0.0, held_for - dt*0.8)  # tiny forgiveness, not too kind

                    if held_for >= hold_sec:
                        success = True
                        break
                else:
                    # trap round: if they match, instant fail
                    if is_match:
                        fail_reason = "YOU OBEYED A LIE."
                        break

                remaining = max(0.0, t_deadline - time.time())
                need = max(0.0, hold_sec - held_for)

                draw_hud(
                    frame, score, streak, detected_groups, detected_comp,
                    msg_top=prompt_text + label,
                    msg_mid=("HOLD IT!" if simon else "DON'T DO IT."),
                    msg_bot=(f"held={held_for:.2f}/{hold_sec:.2f}  need={need:.2f}  left={remaining:.1f}s"
                             if simon else f"left={remaining:.1f}s"),
                    dbg=dbg,
                    color=(0,255,0) if (simon and is_match) else (255,255,255)
                    # --- debug overlay (ARMS) ---
                )
                cv2.imshow(WINDOW_NAME, frame)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    return

                elapsed = time.time() - t0
                sleep_for = FRAME_TIME - elapsed
                if sleep_for > 0:
                    time.sleep(sleep_for)

            # resolve if time ended
            if simon and not success and not fail_reason:
                fail_reason = "TOO SLOW. HUMAN LATENCY DETECTED."

            # scoring
            if fail_reason:
                # show fail screen
                t_end = time.time() + 1.5
                while time.time() < t_end:
                    t0 = time.time()
                    frame = read_frame(jpeg_stream)
                    if frame is None:
                        continue

                    draw_hud(
                        frame, score, streak, detected_groups, detected_comp,
                        msg_top="FAIL",
                        msg_mid=fail_reason,
                        msg_bot=f"Final streak: {streak}",
                        dbg=dbg,    
                        color=(0,0,255)
                    )
                    cv2.imshow(WINDOW_NAME, frame)
                    if (cv2.waitKey(1) & 0xFF) == ord("q"):
                        return

                    elapsed = time.time() - t0
                    sleep_for = FRAME_TIME - elapsed
                    if sleep_for > 0:
                        time.sleep(sleep_for)

                # reset run
                streak = 0
                # optional: soften difficulty back a touch after death
                time_limit = min(TIME_LIMIT_SEC, time_limit / SPEEDUP_FACTOR)
                hold_sec   = min(HOLD_SEC, hold_sec / SPEEDUP_FACTOR)
            else:
                score += 1
                streak += 1

                # speed up
                if streak % SPEEDUP_EVERY == 0:
                    time_limit = max(1.3, time_limit * SPEEDUP_FACTOR)
                    hold_sec   = max(0.45, hold_sec * SPEEDUP_FACTOR)

                # show success flash
                t_end = time.time() + 0.7
                while time.time() < t_end:
                    t0 = time.time()
                    frame = read_frame(jpeg_stream)
                    if frame is None:
                        continue

                    draw_hud(
                        frame, score, streak, detected_groups, detected_comp,
                        msg_top="OK",
                        msg_mid="You may live.",
                        msg_bot=f"Next: limit={time_limit:.1f}s hold={hold_sec:.1f}s",
                        dbg=dbg,
                        color=(0,255,0)
                    )
                    cv2.imshow(WINDOW_NAME, frame)
                    if (cv2.waitKey(1) & 0xFF) == ord("q"):
                        return

                    elapsed = time.time() - t0
                    sleep_for = FRAME_TIME - elapsed
                    if sleep_for > 0:
                        time.sleep(sleep_for)

            time.sleep(COOLDOWN_SEC)

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
