import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

NUM = r"[+-]?(?:\d+\.\d*|\d*\.\d+|\d+)(?:[eE][+-]?\d+)?"

RE_EXPERIMENT = re.compile(r"^\s*EXPERIMENT\s+(\d+)\s*$")
RE_PHASE      = re.compile(r"^\s*(TRAINING|REFINEMENT)\s*$")
RE_EPOCH      = re.compile(r"^\s*Epoch\s+(\d+)\s*$")

RE_TRAIN = re.compile(rf"^\s*training\s+loss\s+({NUM})\s*,\s*\(([^)]*)\)\s*$", re.IGNORECASE)
RE_VALID = re.compile(rf"^\s*validation\s+loss\s+({NUM})\s*,\s*\(([^)]*)\)\s*$", re.IGNORECASE)

def _parse_components(s: str) -> List[float]:
    # "0.025245927, 146.2193, 4.35872, 0.84202135" -> [..floats..]
    parts = [p.strip() for p in s.split(",") if p.strip()]
    out: List[float] = []
    for p in parts:
        # defensive: ignore weird tokens instead of crashing
        try:
            out.append(float(p))
        except ValueError:
            pass
    return out

def parse_training_log(path: Union[str, Path]) -> Dict[int, Any]:
    """
    Parse a TF training console log.

    Returns:
      results[exp_idx]["TRAINING"][epoch]   -> dict with train/valid losses+components
      results[exp_idx]["REFINEMENT"][epoch] -> same
      results[exp_idx]["epochs"][f"{phase}:{epoch}"] -> same (flattened, unique keys)
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="replace").splitlines()

    results: Dict[int, Any] = {}

    cur_exp: Optional[int] = None
    cur_phase: Optional[str] = None
    cur_epoch: Optional[int] = None

    # temp holder until we have both train + valid for the same epoch
    pending: Optional[Dict[str, Any]] = None

    def _ensure(exp: int):
        if exp not in results:
            results[exp] = {"TRAINING": {}, "REFINEMENT": {}, "epochs": {}}

    def _flush_pending():
        nonlocal pending
        if (
            pending is None
            or cur_exp is None
            or cur_phase is None
            or cur_epoch is None
        ):
            pending = None
            return

        # Only store if we have at least one of train/valid (usually both exist)
        if ("train_loss" in pending) or ("valid_loss" in pending):
            _ensure(cur_exp)
            results[cur_exp][cur_phase][cur_epoch] = dict(pending)
            results[cur_exp]["epochs"][f"{cur_phase}:{cur_epoch}"] = dict(pending)

        pending = None

    for line in text:
        m = RE_EXPERIMENT.match(line)
        if m:
            _flush_pending()
            cur_exp = int(m.group(1))
            cur_phase = None
            cur_epoch = None
            pending = None
            _ensure(cur_exp)
            continue

        m = RE_PHASE.match(line)
        if m:
            _flush_pending()
            cur_phase = m.group(1)
            cur_epoch = None
            pending = None
            continue

        m = RE_EPOCH.match(line)
        if m:
            _flush_pending()
            cur_epoch = int(m.group(1))
            pending = {"phase": cur_phase, "epoch": cur_epoch}
            continue

        m = RE_TRAIN.match(line)
        if m and pending is not None:
            pending["train_loss"] = float(m.group(1))
            pending["train_loss_components"] = _parse_components(m.group(2))
            continue

        m = RE_VALID.match(line)
        if m and pending is not None:
            pending["valid_loss"] = float(m.group(1))
            pending["valid_loss_components"] = _parse_components(m.group(2))
            continue

    _flush_pending()
    return results

if __name__ == "__main__":
    # Example usage:
    log_path = "training_validation_losses.txt"
    results = parse_training_log(log_path)

    # Example: get Experiment 0, TRAINING epoch 500
    ex0_e500 = results[0]["TRAINING"][500]
    print(ex0_e500)

    # Optional: save for later
    # import json, pickle
    # json.dump(results, open("losses.json","w"), indent=2)  # (floats OK)
    # pickle.dump(results, open("losses.pkl","wb"))
