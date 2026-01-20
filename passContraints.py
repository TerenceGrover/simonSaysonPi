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