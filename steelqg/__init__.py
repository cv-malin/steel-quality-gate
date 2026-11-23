"""steelqg — 钢材表面缺陷智能质检台核心包"""
from .synth import (DEFECTS, DEFECT_TYPES, DEFECT_CN, SEVERITY_RANK,
                    Defect, Board, synth_board, synth_dataset, board_truth_verdict)
from .detector import Detection, detect_baseline, detect_improved, decide
from .evaluate import run_experiment

__version__ = "1.0.0"
__all__ = [
    "DEFECTS", "DEFECT_TYPES", "DEFECT_CN", "SEVERITY_RANK",
    "Defect", "Board", "synth_board", "synth_dataset", "board_truth_verdict",
    "Detection", "detect_baseline", "detect_improved", "decide", "run_experiment",
]
