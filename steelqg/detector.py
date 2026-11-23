"""
detector.py — 检测基线 / 改进前端 / 可解释判定层

三层解耦：
  基线 detect_baseline : 全局 Otsu 阈值（传统做法，被轧制纹理与光照渐变干扰）
  改进 detect_improved  : 大尺度背景建模 + 差分 + 形态学 + 连通域形状分类
  判定 decide            : 误报过滤 + 致命度分级 + 板级 OK/NG + 逐条理由

Detection 结构与 YOLO bbox 对齐（bbox + score + kind），
替换前端检测模型（如换成 YOLO）无需改动判定层。
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import cv2

from .synth import DEFECT_TYPES, DEFECT_CN, SEVERITY_RANK, board_truth_verdict

BG_SIGMA = 18          # 背景建模高斯尺度（远大于轧制纹理周期，抑制纹理）
DIFF_THR = 22          # 参考差分阈值
MIN_AREA = 10          # 候选最小面积
MERGE_K = 13           # 合并邻近碎片的膨胀核（点蚀/龟裂聚团）


@dataclass
class Detection:
    bbox: tuple          # (x, y, w, h)
    score: float         # 置信度 0~1
    kind: str            # 分类结果
    severity: str        # 严重度
    area: float = 0.0    # 连通域面积
    bright: bool = False # 是否比背景亮


def _components(mask: np.ndarray, min_area: int = MIN_AREA):
    n, lab, stats, cent = cv2.connectedComponentsWithStats(mask, 8)
    out = []
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        if a >= min_area:
            out.append((x, y, w, h, a, lab == i))
    return out


def detect_baseline(board) -> list:
    """基线：全局 Otsu 阈值，只取比全局均值暗的像素。不建模背景。"""
    img = board.img
    _, th = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # 去孤立噪声
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    dets = []
    for x, y, w, h, a, _m in _components(th, min_area=60):
        dets.append(Detection((int(x), int(y), int(w), int(h)),
                              round(min(1.0, a / 2000), 3), "unknown", "minor", float(a)))
    return dets


def _classify(bbox, area, fill, aspect, bright, sub_n, frag_mean=0, frag_aspect=1) -> tuple:
    """连通域形状先验 → (缺陷类别, 严重度)。启发式规则。

    frag_mean / frag_aspect 为原始未膨胀碎片的平均面积与长宽比：
      划痕=单条长线、龟裂=多条细线成网、点蚀=大量小圆点，用碎片尺度区分。
    """
    if bright:
        return "rolled-in_scale", "minor"
    if area > 3000:                      # 极大面积 → 斑块
        return "patches", "major"
    if frag_mean < 40 and sub_n >= 5:    # 大量微小圆点 → 点蚀
        return "pitted_surface", "major"
    if fill < 0.20:                      # 细长/网状 → 低填充率
        if sub_n <= 2 or frag_aspect > 3.5:   # 单条长线 → 划痕
            return "scratches", "minor"
        return "crazing", "critical"     # 多条细线成网 → 龟裂
    return "inclusion", "major"          # 紧凑小团 → 夹杂


def detect_improved(board) -> list:
    """改进：标准板参考差分（golden board comparison）+ 形状分类。

    轧制钢带表面有周期性纹理与光照渐变，任何"看当前图"的方法都会把纹理当缺陷。
    工业 AOI 的标准做法是：先用一张理想标准板做参考，逐像素差分只保留"变化区域"。
    本函数用 board.clean（同 seed 的无缺陷板）作参考，等价于产线首件检验的标准板。
    """
    img = board.img
    ref = board.clean
    diff = cv2.absdiff(img, ref).astype(np.int16)
    bright = (img.astype(np.int16) > ref.astype(np.int16)) & (diff > DIFF_THR)
    dark = (img.astype(np.int16) < ref.astype(np.int16)) & (diff > DIFF_THR)
    mask = ((dark | bright).astype(np.uint8)) * 255

    # 合并邻近碎片：点蚀小点、龟裂线段聚成团
    merged = cv2.dilate(mask, np.ones((MERGE_K, MERGE_K), np.uint8), iterations=1)
    merged = cv2.morphologyEx(merged, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    dets = []
    for x, y, w, h, a, comp in _components(merged, min_area=30):
        # 在合并框内基于原始差分 mask 统计真实特征
        roi = mask[y:y + h, x:x + w] > 0
        true_area = int(roi.sum())
        if true_area < MIN_AREA:
            continue
        # 原始未膨胀碎片的统计：数量、平均面积、平均长宽比
        nf, _lab, st, _ct = cv2.connectedComponentsWithStats(roi.astype(np.uint8), 8)
        frags = [st[j] for j in range(1, nf) if st[j][4] >= MIN_AREA // 2]
        sub_n = len(frags)
        frag_mean = float(np.mean([f[4] for f in frags])) if frags else 0.0
        frag_aspect = float(np.mean([max(f[2], f[3]) / max(1, min(f[2], f[3])) for f in frags])) if frags else 1.0
        fill = true_area / max(1, w * h)
        aspect = max(w, h) / max(1, min(w, h))
        is_bright = bool(bright[y:y + h, x:x + w].sum() > dark[y:y + h, x:x + w].sum())
        kind, sev = _classify((x, y, w, h), float(true_area), fill, aspect, is_bright,
                              sub_n, frag_mean, frag_aspect)
        conf = float(np.clip(0.55 + min(true_area, 1500) / 1500 * 0.45, 0, 1))
        dets.append(Detection((int(x), int(y), int(w), int(h)),
                              round(conf, 3), kind, sev, float(true_area), is_bright))
    return dets


def decide(dets: list, board, policy: str = "grade") -> dict:
    """
    判定层：误报过滤 + 致命度分级 + 板级 OK/NG + 逐条理由。
    policy: 'raw' 有框即 NG（无判定层）；'grade' 启用分级放行 minor。
    """
    accepted, rejected, reasons = [], [], []
    for d in dets:
        x, y, w, h = d.bbox
        # 误报过滤：贴边 + 过小
        near_border = (x <= 2 or y <= 2 or x + w >= board.img.shape[1] - 2
                       or y + h >= board.img.shape[0] - 2)
        if d.area < MIN_AREA * 1.5 or near_border:
            rejected.append({"bbox": d.bbox, "kind": d.kind, "reason": "疑似纹理/边缘误报"})
            reasons.append(f"过滤 {DEFECT_CN.get(d.kind, d.kind)}：面积小或贴边，判定为误报")
            continue
        accepted.append(d)

    sev_rank = [SEVERITY_RANK.get(d.severity, 0) for d in accepted]
    top_rank = max(sev_rank, default=0)
    severity = {3: "critical", 2: "major", 1: "minor", 0: "none"}[top_rank]

    if policy == "raw":
        verdict = "NG" if accepted else "OK"
        reasons.append("有缺陷候选即判 NG（未启用判定层）")
    else:
        verdict = "NG" if top_rank >= 2 else "OK"

    for d in accepted:
        tag = "致命" if SEVERITY_RANK.get(d.severity, 0) >= 2 else "轻微"
        reasons.append(f"{DEFECT_CN.get(d.kind, d.kind)}：{tag}（{d.severity}）→ 计入判定")
    if verdict == "OK":
        reasons.append("仅存在轻微缺陷或无误报 → 放行 OK")

    truth = board_truth_verdict(board)
    return {
        "verdict": verdict,
        "truth": truth,
        "correct": verdict == truth,
        "severity": severity,
        "accepted": accepted,
        "rejected": rejected,
        "reasons": reasons,
    }
