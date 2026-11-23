"""
evaluate.py — 三档对照评测（缺陷级 + 板级 + 严重度级）

三档：
  baseline        全局阈值
  improved        背景差分（不分级：有框即 NG）
  improved_grade  背景差分 + 判定层（过滤 + 分级放行 minor）

IoU 命中阈值三档一致，保证对照公平。
"""
from __future__ import annotations

from .synth import synth_dataset, DEFECT_TYPES, board_truth_verdict, SEVERITY_RANK
from .detector import detect_baseline, detect_improved, decide

IOU_THR = 0.15


def iou(a, b) -> float:
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[0] + a[2], b[0] + b[2]); y2 = min(a[1] + a[3], b[1] + b[3])
    iw, ih = max(0, x2 - x1), max(0, y2 - y1)
    inter = iw * ih
    u = a[2] * a[3] + b[2] * b[3] - inter
    return inter / u if u > 0 else 0.0


def _matched(dets, board):
    """返回 (命中数, 命中检测列表)，一框只匹配一个真值。"""
    gt = [d.bbox for d in board.defects]
    used = set()
    hit = 0
    hit_dets = []
    for d in dets:
        best, bi = IOU_THR, -1
        for i, g in enumerate(gt):
            if i in used:
                continue
            v = iou(d.bbox, g)
            if v > best:
                best, bi = v, i
        if bi >= 0:
            used.add(bi)
            hit += 1
            hit_dets.append(d)
    return hit, hit_dets


def _metrics(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": round(p, 4),
            "recall": round(r, 4), "f1": round(f1, 4)}


def run_experiment(n_per_type: int = 20, n_clean: int = 20,
                   difficulty: str = "normal", seed0: int = 1000) -> dict:
    boards = synth_dataset(n_per_type, n_clean, difficulty, seed0)
    modes = ["baseline", "improved", "improved_grade"]

    # 缺陷级累计
    acc = {m: {"tp": 0, "fp": 0, "fn": 0} for m in modes}
    per_class = {m: {k: {"hit": 0, "total": 0} for k in DEFECT_TYPES} for m in modes}
    # 板级
    board_acc = {m: 0 for m in modes}
    fn_board = {m: 0 for m in modes}
    fp_board = {m: 0 for m in modes}
    # 严重度级
    sev_correct = {m: 0 for m in modes}
    sev_total = 0

    for b in boards:
        truth_v = board_truth_verdict(b)
        truth_sev = max((d.severity for d in b.defects),
                        key=lambda s: SEVERITY_RANK.get(s, 0), default="none")
        if b.defects:
            sev_total += 1

        db = detect_baseline(b)
        di = detect_improved(b)
        results = {
            "baseline": db,
            "improved": di,
        }
        dec = decide(di, b, policy="raw")
        dec_g = decide(di, b, policy="grade")
        results["improved_grade"] = di

        verdicts = {
            "baseline": "NG" if db else "OK",
            "improved": dec["verdict"],
            "improved_grade": dec_g["verdict"],
        }
        sevs = {
            "baseline": "none",
            "improved": dec["severity"],
            "improved_grade": dec_g["severity"],
        }

        for m in modes:
            dets = results[m]
            hit, _ = _matched(dets, b)
            gt_n = len(b.defects)
            acc[m]["tp"] += hit
            acc[m]["fp"] += len(dets) - hit
            acc[m]["fn"] += gt_n - hit
            if gt_n:
                k = b.kind
                per_class[m][k]["total"] += 1
                if hit > 0:
                    per_class[m][k]["hit"] += 1

            v = verdicts[m]
            if v == truth_v:
                board_acc[m] += 1
            if truth_v == "NG" and v == "OK":
                fn_board[m] += 1
            if truth_v == "OK" and v == "NG":
                fp_board[m] += 1
            if b.defects and sevs[m] == truth_sev:
                sev_correct[m] += 1

    total = len(boards)
    out = {
        "config": {
            "n_per_type": n_per_type, "n_clean": n_clean,
            "difficulty": difficulty, "seed0": seed0,
            "total_boards": total, "iou_thr": IOU_THR,
        },
        "results": [],
    }
    for m in modes:
        det = _metrics(**acc[m])
        out["results"].append({
            "mode": m,
            **det,
            "board_acc": round(board_acc[m] / total, 4),
            "fn_board": fn_board[m],
            "fp_board": fp_board[m],
            "sev_acc": round(sev_correct[m] / sev_total, 4) if sev_total else 0.0,
            "per_class_recall": {
                k: round(v["hit"] / v["total"], 4) if v["total"] else 0.0
                for k, v in per_class[m].items()
            },
        })
    return out
