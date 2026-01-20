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
import requests
import mediapipe as mp

from audioManager import play_prompt_audio
from loadHelpers import (load_json, sanitize_constraints)
from mjpegDecoder import mjpeg_frames_from_pipe
from passContraints import passes_constraints
from hudDraw import draw_hud

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
VIS_WRIST = 0.1
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
# LOADING POSES + COMPOUNDS
# ============================================================
POSES_BY_GROUP = {"arms": [], "legs": [], "torso": []}
COMPOUNDS = {}

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

    for p in POSES_BY_GROUP[group_name]:
        if not pose_visibility_ok(vis_map, p):
            continue
        if not passes_constraints(feats, p):
            continue

        sc = score_pose(feats, p)
        if sc < best_score:
            second_score = best_score
            best_score = sc
            best_name = p["pose_name"]
        elif sc < second_score:
            second_score = sc

    if best_score >= 1e8:
        return "UNKNOWN"

    if best_score > MAX_ACCEPTABLE_SCORE[group_name]:
        return "UNKNOWN"
    if (second_score - best_score) < MARGIN_MIN[group_name]:
        return "UNKNOWN"

    return best_name

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
        feats, vis_map = extract_features_and_vismap(pose_landmarks.landmark)

        for g in ["arms", "legs", "torso"]:
            name = detect_group(feats, vis_map, g)
            name = self._smooth(g, name)
            detected[g] = name

        # Decide compound if any matches (greedy: first match)
        compound = None
        for cname in COMPOUNDS.keys():
            if matches_compound(detected, cname):
                compound = cname
                break

        return detected, compound

# ============================================================
# GAME: build a list of playable prompts
# ============================================================
def build_prompt_pool():
    """
    We'll build prompts from:
      - atomic presets: any single group pose (arms_* / legs_* / torso_*)
      - compound presets: each compound by name
    """
    atomic = []
    for g in ["arms", "legs", "torso"]:
        for p in POSES_BY_GROUP[g]:
            atomic.append(("atomic", p["pose_name"]))

    compounds = [("compound", name) for name in COMPOUNDS.keys()]
    pool = atomic + compounds

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

# ============================================================
# STATEFUL SIMON SAYS MECHANIC (lock state until released)
# ============================================================

ARMS_NEUTRAL = "UNKNOWN"     # we accept UNKNOWN as "hands down"
LEGS_NEUTRAL = "UNKNOWN"     # standing
TORSO_NEUTRAL = "UNKNOWN"

# Map detected pose_name -> abstract substate (only where we need it)
# NOTE: You can extend these if you have more atomic unilateral arm poses later.
def obs_from_detected(detected_groups):
    """
    Convert detected_groups (pose names or UNKNOWN) into a simplified observation.
    We keep it mostly as pose_name per group, because your classifier already
    outputs pose_name strings.
    """
    return {
        "arms": detected_groups.get("arms", "UNKNOWN"),
        "legs": detected_groups.get("legs", "UNKNOWN"),
        "torso": detected_groups.get("torso", "UNKNOWN"),
    }

def poses_match(expected_pose_name, observed_pose_name, group):
    """
    Matching rule with your "hands down == UNKNOWN" hack.
    """
    if group == "arms":
        # arms_hands_down is unreliable -> treat UNKNOWN as "down"
        if expected_pose_name in ("arms_hands_down", "UNKNOWN"):
            return observed_pose_name == "UNKNOWN"
    if group == "legs":
        if expected_pose_name in ("legs_stand_up", "UNKNOWN"):
            return observed_pose_name == "UNKNOWN"
    # Default: strict match
    return observed_pose_name == expected_pose_name

def obs_matches_state(obs, state, groups=("arms","legs","torso")):
    for g in groups:
        if not poses_match(state[g], obs[g], g):
            return False
    return True

class GameState:
    """
    The current "locked" expected posture. This is what the player MUST maintain
    unless Simon Says updates it.
    """
    def __init__(self):
        self.state = {"arms": "UNKNOWN", "legs": "UNKNOWN", "torso": "UNKNOWN"}

    def copy(self):
        s = GameState()
        s.state = dict(self.state)
        return s

    def apply(self, next_state_dict):
        self.state.update(next_state_dict)

# ============================================================
# STRICT COMMAND SYSTEM (HARD LOCK, NO ILLEGAL PICKS)
# ============================================================

RETURN_BOOST = 6.0   # make releases common, avoids softlocks
BASE_W = 1.0

GROUPS = ("arms", "legs", "torso")

class Command:
    def __init__(self, name, affected_groups, apply_fn, weight_fn=None):
        self.name = name
        self.affected_groups = set(affected_groups)
        self.apply_fn = apply_fn
        self.weight_fn = weight_fn or (lambda st: BASE_W)

    def apply(self, st):
        return self.apply_fn(st)

    def weight(self, st):
        return float(self.weight_fn(st))


def clamp_w(w):
    try:
        return max(0.0, float(w))
    except Exception:
        return 0.0


def build_commands():
    cmds = []

    # ---------- ARMS ----------
    # set
    cmds.append(Command(
        name="arms_hands_up",
        affected_groups=("arms",),
        apply_fn=lambda st: {"arms": "arms_hands_up"},
        weight_fn=lambda st: 0.0 if st["arms"] == "arms_hands_up" else BASE_W
    ))

    # release (always available when locked)
    cmds.append(Command(
        name="arms_hands_down",
        affected_groups=("arms",),
        apply_fn=lambda st: {"arms": "UNKNOWN"},
        weight_fn=lambda st: (RETURN_BOOST if st["arms"] != "UNKNOWN" else 0.0)
    ))

    # (Optional) other arm poses: only reachable from neutral, released back to neutral
    for pn in ["arms_t_pose", "arms_cross_arms", "arms_touch_nose", "arms_hand_on_head"]:
        cmds.append(Command(
            name=pn,
            affected_groups=("arms",),
            apply_fn=lambda st, _pn=pn: {"arms": _pn},
            weight_fn=lambda st: (0.6 if st["arms"] == "UNKNOWN" else 0.0)
        ))

    # ---------- LEGS ----------
    for lp in ["legs_split", "legs_squat", "legs_single_leg_up_L", "legs_single_leg_up_R"]:
        cmds.append(Command(
            name=lp,
            affected_groups=("legs",),
            apply_fn=lambda st, _lp=lp: {"legs": _lp},
            weight_fn=lambda st: (BASE_W if st["legs"] == "UNKNOWN" else 0.0)  # ONLY from neutral
        ))

    cmds.append(Command(
        name="legs_stand_up",
        affected_groups=("legs",),
        apply_fn=lambda st: {"legs": "UNKNOWN"},
        weight_fn=lambda st: (RETURN_BOOST if st["legs"] != "UNKNOWN" else 0.0)
    ))

    # ---------- TORSO (if you have it) ----------
    # Same pattern as legs: only from neutral, and a torso_neutral release command.

    return cmds


COMMANDS = build_commands()


def _is_legal_transition(cur, nxt):
    """
    HARD RULE:
      - If cur == UNKNOWN: nxt can be UNKNOWN or a pose (set)
      - If cur != UNKNOWN: nxt MUST be UNKNOWN (release only)
    """
    if cur == "UNKNOWN":
        return True  # staying UNKNOWN or setting a pose is allowed
    # locked in pose -> only release allowed
    return (nxt == "UNKNOWN")


def _command_is_legal_for_state(cmd, st):
    applied = cmd.apply(st)
    for g in cmd.affected_groups:
        cur = st[g]
        nxt = applied.get(g, cur)
        if not _is_legal_transition(cur, nxt):
            return False
        # ALSO ban "pose -> different pose" explicitly
        if cur != "UNKNOWN" and nxt != "UNKNOWN":
            return False
    return True


def weighted_choice(items, weights):
    total = sum(weights)
    if total <= 0:
        return None
    r = random.random() * total
    acc = 0.0
    for it, w in zip(items, weights):
        acc += w
        if r <= acc:
            return it
    return items[-1]


def pick_next_command(current_state):
    """
    This picker cannot output illegal commands.
    It enforces:
      - If a group is locked, only its release command is selectable.
      - Legs: no pose->pose ever, must stand_up between.
      - Release commands get boosted weight to prevent softlocks.
    """
    st = current_state

    valid = []
    weights = []

    for c in COMMANDS:
        print(f"Checking command {c.name} for state {st}")
        if not _command_is_legal_for_state(c, st):
            continue
        w = clamp_w(c.weight(st))
        if w <= 0:
            continue
        valid.append(c)
        weights.append(w)

    # If nothing is valid, you are in a contradictory state — force releases
    if not valid:
        # force-release priority: legs, arms, torso
        for release in ("legs_stand_up", "arms_hands_down"):
            for c in COMMANDS:
                if c.name == release:
                    return c
        return None

    return weighted_choice(valid, weights)


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

    time.sleep(1.0)  # let camera settle
    try:
        while True:
            # pick target
            # --- stateful selection ---
            if "game_state" not in locals():
                game_state = GameState()  # locked expected posture

            current_locked = dict(game_state.state)

            cmd = pick_next_command(current_locked)
            if cmd is None:
              continue  # no legal move, skip this tick
            target_name = cmd.name
            target_kind = "atomic"  # treat commands as atomic actions

            # Simon says vs trap (trap must be common too)
            simon = (random.random() < SIMON_SAYS_PROB)

            # If simon: expected state will change to the command-applied state
            # If trap: expected state stays the same
            next_state = dict(current_locked)
            next_state.update(cmd.apply(current_locked))

            print(f"Next command: {'SIMON SAYS' if simon else 'TRAP'} -> {target_name} | ")
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
                    detected_groups, detected_comp = detector.detect(res.pose_landmarks)

                draw_hud(
                    frame, score, streak, detected_groups, detected_comp,
                    msg_top=prompt_text + label,
                    msg_mid="Get ready...",
                    msg_bot=f"limit={time_limit:.1f}s hold={hold_sec:.1f}s",
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

                obs = obs_from_detected(detected_groups)

                # ========= HARD LOCK STATE VALIDATION =========

                for g in ("arms", "legs", "torso"):
                    cur = current_locked[g]      # Simon-locked state
                    seen = obs[g]                # what the player is doing
                    target = next_state[g]       # state AFTER command (if Simon)

                    if g not in cmd.affected_groups:
                        # This group is NOT being commanded
                        if seen != cur:
                            fail_reason = "YOU MOVED WITHOUT PERMISSION."
                            break

                    else:
                        # This group IS being commanded
                        if not simon:
                            # Trap: group must not change at all
                            if seen != cur:
                                fail_reason = "YOU OBEYED A LIE."
                                break

                        else:
                            # Simon Says — HARD LOCK LOGIC
                            if cur == "UNKNOWN":
                                # Only legal move: UNKNOWN → target
                                if seen not in ("UNKNOWN", target):
                                    fail_reason = "ILLEGAL TRANSITION."
                                    break
                            else:
                                # Only legal move: cur → UNKNOWN (release)
                                if target != "UNKNOWN":
                                    fail_reason = "ILLEGAL TRANSITION."
                                    break
                                if seen not in (cur, "UNKNOWN"):
                                    fail_reason = "ILLEGAL TRANSITION."
                                    break



                # Trap round: even affected group must NOT become the commanded next_state
                if not simon:
                    if obs_matches_state(obs, next_state, groups=cmd.affected_groups):
                        fail_reason = "YOU OBEYED A LIE."
                        break
                    # In trap, also require affected group to remain locked
                    if not obs_matches_state(obs, current_locked, groups=cmd.affected_groups):
                        fail_reason = "YOU MOVED WITHOUT PERMISSION."
                        break
                else:
                    # Simon round: success when affected group reaches next_state and holds
                    if obs_matches_state(obs, next_state, groups=cmd.affected_groups):
                        held_for += dt
                    else:
                        held_for = max(0.0, held_for - dt*0.8)

                    if held_for >= hold_sec:
                        success = True
                        break

                remaining = max(0.0, t_deadline - time.time())
                need = max(0.0, hold_sec - held_for)

                draw_hud(
                    frame, score, streak, detected_groups, detected_comp,
                    msg_top=prompt_text + label,
                    msg_mid=("HOLD IT!" if simon else "DON'T DO IT."),
                    msg_bot=(f"held={held_for:.2f}/{hold_sec:.2f}  need={need:.2f}  left={remaining:.1f}s"
                             if simon else f"left={remaining:.1f}s"),
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
                # Ping http://10.0.0.148/release
                requests.get("http://10.0.0.148/release")
                #fucking crash the whole thing just break
                break 
            
            else:
                if simon:
                    game_state.apply(cmd.apply(game_state.state))
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