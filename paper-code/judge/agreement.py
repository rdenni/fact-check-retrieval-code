"""agreement.py -- Cohen's kappa and friends, shared by adjudicate.py and validate_judge.py."""
from __future__ import annotations


def cohen_kappa(pairs: list[tuple[str, str]]) -> dict:
    """pairs of (label_a, label_b) with labels 'y'/'n'. Returns kappa and the confusion counts."""
    n = len(pairs)
    if n == 0:
        return {"n": 0, "kappa": None, "observed": None, "expected": None}
    yy = sum(1 for a, b in pairs if a == "y" and b == "y")
    yn = sum(1 for a, b in pairs if a == "y" and b == "n")
    ny = sum(1 for a, b in pairs if a == "n" and b == "y")
    nn = sum(1 for a, b in pairs if a == "n" and b == "n")
    po = (yy + nn) / n
    pa, pb = (yy + yn) / n, (yy + ny) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    kappa = (po - pe) / (1 - pe) if (1 - pe) > 0 else 1.0
    return {"n": n, "kappa": round(kappa, 4), "observed": round(po, 4), "expected": round(pe, 4),
            "yy": yy, "yn": yn, "ny": ny, "nn": nn, "pos_rate_a": round(pa, 4), "pos_rate_b": round(pb, 4)}


def prf(pairs: list[tuple[str, str]]) -> dict:
    """pairs of (predicted, gold): precision, recall, F1 of the 'y' class."""
    tp = sum(1 for p, g in pairs if p == "y" and g == "y")
    fp = sum(1 for p, g in pairs if p == "y" and g == "n")
    fn = sum(1 for p, g in pairs if p == "n" and g == "y")
    prec = tp / (tp + fp) if tp + fp else None
    rec = tp / (tp + fn) if tp + fn else None
    f1 = 2 * prec * rec / (prec + rec) if prec and rec else (0.0 if prec is not None and rec is not None else None)
    return {"precision": None if prec is None else round(prec, 4), "recall": None if rec is None else round(rec, 4),
            "f1": None if f1 is None else round(f1, 4), "tp": tp, "fp": fp, "fn": fn}
