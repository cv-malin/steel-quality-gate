"""
synth.py — 程序化钢带表面样本合成器（可复现，带真值）

对齐东北大学 NEU-DET 数据集的 6 类热轧钢带表面缺陷：
  crazing(龟裂) / inclusion(夹杂) / patches(斑块) /
  pitted_surface(点蚀) / rolled-in_scale(轧制鳞片) / scratches(划痕)

合成动机：真实 NEU-DET 需申请，且原始缺陷本身为人工合成标注。
本合成器用 seed 控制，提供等价、可复现、带真值的实验条件。

关键设计：背景带"水平轧制纹理 + 光照渐变"，使全局阈值基线必然失败，
从而构造出与传统方法可对照的结构性差异（这是本作品对照实验的基础）。
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import cv2

# NEU-DET 六类缺陷：key -> (中文名, 严重度, 相对对比度, 尺寸基准px)
DEFECTS = {
    "crazing":         ("龟裂",     "critical", 0.55, 70),
    "inclusion":       ("夹杂",     "major",    0.85, 42),
    "patches":         ("斑块",     "major",    0.75, 95),
    "pitted_surface":  ("点蚀",     "major",    0.50, 60),
    "rolled-in_scale": ("轧制鳞片", "minor",    0.32, 80),
    "scratches":       ("划痕",     "minor",    0.30, 110),
}
DEFECT_TYPES = list(DEFECTS)
DEFECT_CN = {k: v[0] for k, v in DEFECTS.items()}
SEVERITY_RANK = {"minor": 1, "major": 2, "critical": 3}

SIZE = 640
_SCALE = {"easy": 1.25, "normal": 1.0, "hard": 0.62}


@dataclass
class Defect:
    kind: str
    bbox: tuple          # (x, y, w, h)
    severity: str
    contrast: float


@dataclass
class Board:
    img: np.ndarray      # 灰度 uint8 (H, W)
    clean: np.ndarray    # 无缺陷的同一块板（用于改进前端的参考建模验证）
    defects: list        # List[Defect]
    seed: int
    kind: str
    difficulty: str


def _background(rng: np.random.Generator, size: int = SIZE) -> np.ndarray:
    """带轧制纹理与光照渐变的钢带背景。"""
    h = w = size
    # 缓慢光照渐变（模拟现场不均匀照明）
    grad = np.linspace(-22, 22, w)[None, :] + np.linspace(-12, 12, h)[:, None]
    # 水平轧制纹理：多频周期条纹（这是全局阈值失效的根源）
    y = np.arange(h, dtype=np.float32)
    texture = 7 * np.sin(y / 3.1) + 4 * np.sin(y / 7.7 + 0.9) + 3 * np.sin(y / 17.0)
    texture = texture[:, None]
    # 细颗粒噪声
    noise = rng.normal(0, 3.5, (h, w))
    base = 132.0 + grad + texture + noise
    return np.clip(base, 0, 255).astype(np.uint8)


def _put_defect(img: np.ndarray, kind: str, rng: np.random.Generator,
                scale: float) -> tuple:
    """在 img 上绘制单类缺陷，返回 (x, y, w, h) bbox。"""
    h, w = img.shape
    contrast_scaled = DEFECTS[kind][2] * (1.0 if scale >= 1.0 else 0.75 + 0.25 * scale)
    dark = int(255 * 0.62 * contrast_scaled)   # 缺陷相对背景的暗度

    if kind == "crazing":  # 龟裂：网状细裂纹
        cx, cy = rng.integers(w * 0.3, w * 0.7), rng.integers(h * 0.3, h * 0.7)
        extent = int(DEFECTS[kind][3] * scale)
        for _ in range(rng.integers(6, 11)):
            x0, y0 = cx + rng.integers(-extent // 2, extent // 2), cy + rng.integers(-extent // 2, extent // 2)
            pts = [(x0, y0)]
            for _ in range(rng.integers(3, 6)):
                x0 += rng.integers(-extent // 5, extent // 5)
                y0 += rng.integers(-extent // 5, extent // 5)
                pts.append((x0, y0))
            cv2.polylines(img, [np.array(pts, np.int32)], False, int(128 - dark), 1, cv2.LINE_AA)
        R = extent // 2 + 6
        return (int(cx - R), int(cy - R), int(2 * R), int(2 * R))

    if kind == "inclusion":  # 夹杂：不规则深色团块
        cx, cy = rng.integers(w * 0.25, w * 0.75), rng.integers(h * 0.25, h * 0.75)
        r = int(DEFECTS[kind][3] * scale * rng.uniform(0.8, 1.2)) // 2
        pts = np.array([[cx + r * rng.uniform(0.6, 1.2) * np.cos(a),
                         cy + r * rng.uniform(0.6, 1.2) * np.sin(a)]
                        for a in np.linspace(0, 2 * np.pi, 12)], np.int32)
        cv2.fillPoly(img, [pts], int(128 - dark))
        x, y, bw, bh = cv2.boundingRect(pts)
        return (x, y, bw, bh)

    if kind == "patches":  # 斑块：大面积浅/深色区域
        cx, cy = rng.integers(w * 0.25, w * 0.75), rng.integers(h * 0.25, h * 0.75)
        r = int(DEFECTS[kind][3] * scale) // 2
        pts = np.array([[cx + r * rng.uniform(0.7, 1.25) * np.cos(a),
                         cy + r * rng.uniform(0.7, 1.25) * np.sin(a)]
                        for a in np.linspace(0, 2 * np.pi, 16)], np.int32)
        cv2.fillPoly(img, [pts], int(128 - dark))
        x, y, bw, bh = cv2.boundingRect(pts)
        return (x, y, bw, bh)

    if kind == "pitted_surface":  # 点蚀：密集小暗点群
        cx, cy = rng.integers(w * 0.3, w * 0.7), rng.integers(h * 0.3, h * 0.7)
        extent = int(DEFECTS[kind][3] * scale)
        for _ in range(rng.integers(30, 55)):
            px = int(np.clip(cx + rng.integers(-extent // 2, extent // 2), 2, w - 3))
            py = int(np.clip(cy + rng.integers(-extent // 2, extent // 2), 2, h - 3))
            cv2.circle(img, (px, py), rng.integers(1, 3), int(128 - dark), -1)
        R = extent // 2 + 4
        return (int(cx - R), int(cy - R), int(2 * R), int(2 * R))

    if kind == "rolled-in_scale":  # 轧制鳞片：沿轧制方向拉长的浅色鳞片带
        cx, cy = rng.integers(w * 0.25, w * 0.75), rng.integers(h * 0.3, h * 0.7)
        length = int(DEFECTS[kind][3] * scale * rng.uniform(0.9, 1.3))
        thick = rng.integers(6, 12)
        x0 = int(np.clip(cx - length // 2, 2, w - length - 2))
        y0 = int(np.clip(cy, 2, h - thick - 2))
        cv2.ellipse(img, (x0 + length // 2, y0 + thick // 2),
                    (length // 2, thick // 2), 0, 0, 360,
                    int(128 + dark * 0.8), -1)
        return (x0, y0, length, thick)

    if kind == "scratches":  # 划痕：细长直线（方向随机，多为斜向）
        cx, cy = rng.integers(w * 0.25, w * 0.75), rng.integers(h * 0.25, h * 0.75)
        length = int(DEFECTS[kind][3] * scale * rng.uniform(0.8, 1.2))
        ang = rng.uniform(0, np.pi)
        dx, dy = int(length * np.cos(ang) / 2), int(length * np.sin(ang) / 2)
        p1 = (int(np.clip(cx - dx, 1, w - 2)), int(np.clip(cy - dy, 1, h - 2)))
        p2 = (int(np.clip(cx + dx, 1, w - 2)), int(np.clip(cy + dy, 1, h - 2)))
        cv2.line(img, p1, p2, int(128 - dark), rng.integers(1, 3), cv2.LINE_AA)
        x, y, bw, bh = cv2.boundingRect(np.array([p1, p2], np.int32))
        return (x - 2, y - 2, bw + 4, bh + 4)

    raise ValueError(f"unknown defect kind: {kind}")


def synth_board(seed: int, kind: str = "short", difficulty: str = "normal",
                size: int = SIZE) -> Board:
    """
    生成一块板。kind 为具体缺陷 key，或 'clean'（良板）。
    difficulty: easy/normal/hard，控制缺陷尺寸。
    """
    if kind not in DEFECT_TYPES and kind != "clean":
        raise ValueError(f"kind must be one of {DEFECT_TYPES} or 'clean'")
    if difficulty not in _SCALE:
        raise ValueError(f"difficulty must be one of {list(_SCALE)}")
    rng = np.random.default_rng(seed)
    clean = _background(rng, size)
    img = clean.copy()
    defects = []
    if kind != "clean":
        bbox = _put_defect(img, kind, rng, _SCALE[difficulty])
        cn, sev, contrast, _ = DEFECTS[kind]
        defects = [Defect(kind, bbox, sev, contrast)]
    return Board(img=img, clean=clean, defects=defects,
                 seed=seed, kind=kind, difficulty=difficulty)


def synth_dataset(n_per_type: int = 20, n_clean: int = 20,
                  difficulty: str = "normal", seed0: int = 1000,
                  size: int = SIZE) -> list:
    """生成完整数据集：6 类各 n_per_type 块 + n_clean 块良板。"""
    out = []
    s = seed0
    for k in DEFECT_TYPES:
        for _ in range(n_per_type):
            out.append(synth_board(s, k, difficulty, size))
            s += 1
    for _ in range(n_clean):
        out.append(synth_board(s, "clean", difficulty, size))
        s += 1
    return out


def board_truth_verdict(board: Board) -> str:
    """真值板级判定：最高严重度 >= major 判 NG。"""
    top = max((d.severity for d in board.defects),
              key=lambda s: SEVERITY_RANK.get(s, 0), default="none")
    return "NG" if SEVERITY_RANK.get(top, 0) >= 2 else "OK"
