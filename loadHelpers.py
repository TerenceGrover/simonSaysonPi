import json
from pathlib import Path

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