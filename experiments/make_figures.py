"""make_figures.py — 从 results.json 生成答辩用三张图。"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

R = json.load(open(os.path.join(HERE, "results.json"), encoding="utf-8"))
res = {x["mode"]: x for x in R["results"]}
MODES = ["baseline", "improved", "improved_grade"]
LABELS = ["全局阈值\n(基线)", "标准板差分\n(改进)", "改进+判定层\n(本作品)"]
COLORS = ["#98a2b3", "#4f8fc0", "#1f6e4c"]

# 图1：三档核心指标
fig, ax = plt.subplots(figsize=(9, 5.2))
x = np.arange(3)
w = 0.26
for i, (key, name) in enumerate([("board_acc", "板级准确率"), ("recall", "缺陷召回率"),
                                  ("precision", "缺陷精确率")]):
    vals = [res[m][key] * 100 for m in MODES]
    bars = ax.bar(x + (i - 1) * w, vals, w, label=name, color=["#b8c4d4", "#e0a94f", "#4f8fc0"][i])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.0f}", ha="center", fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(LABELS, fontsize=11)
ax.set_ylabel("百分比 (%)"); ax.set_ylim(0, 112)
ax.set_title("三档对照：检出召回拉满，判定层消除过杀", fontsize=14, fontweight="bold")
ax.legend(loc="upper left"); ax.grid(axis="y", alpha=.3)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig1_compare.png"), dpi=150); plt.close(fig)

# 图2：误判板数量
fig, ax = plt.subplots(figsize=(8, 4.8))
fn = [res[m]["fn_board"] for m in MODES]
fp = [res[m]["fp_board"] for m in MODES]
ax.bar(x - 0.18, fn, 0.36, label="漏检板 (FN)", color="#c0554d")
ax.bar(x + 0.18, fp, 0.36, label="误报板 (FP)", color="#d99a45")
for i, (a, b) in enumerate(zip(fn, fp)):
    ax.text(i - 0.18, a + 1, str(a), ha="center", fontsize=10)
    ax.text(i + 0.18, b + 1, str(b), ha="center", fontsize=10)
ax.set_xticks(x); ax.set_xticklabels(LABELS, fontsize=11)
ax.set_ylabel("板数 (共140块)"); ax.set_title("板级误判：判定层把误报从 40 块压到 3 块", fontsize=14, fontweight="bold")
ax.legend(); ax.grid(axis="y", alpha=.3)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig2_errors.png"), dpi=150); plt.close(fig)

# 图3：分类型召回
fig, ax = plt.subplots(figsize=(10, 5))
classes = list(res[MODES[0]]["per_class_recall"].keys())
cn = {"crazing":"龟裂", "inclusion":"夹杂", "patches":"斑块",
      "pitted_surface":"点蚀", "rolled-in_scale":"鳞片", "scratches":"划痕"}
xc = np.arange(len(classes)); wc = 0.26
for i, (m, c) in enumerate(zip(MODES, COLORS)):
    vals = [res[m]["per_class_recall"][k] * 100 for k in classes]
    ax.bar(xc + (i - 1) * wc, vals, wc, label=LABELS[i].replace("\n", " "), color=c)
ax.set_xticks(xc); ax.set_xticklabels([cn[k] for k in classes], fontsize=11)
ax.set_ylabel("召回率 (%)"); ax.set_ylim(0, 112)
ax.set_title("分类型召回：基线对 4 类结构性盲区，改进后六类全检出", fontsize=14, fontweight="bold")
ax.legend(); ax.grid(axis="y", alpha=.3)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig3_per_class.png"), dpi=150); plt.close(fig)

print("figures written to", FIG)
