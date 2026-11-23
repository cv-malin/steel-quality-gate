"""steelqg 单元测试：合成器 / 检测器 / 判定层 / 对照实验"""
import unittest
import numpy as np

from steelqg.synth import (DEFECT_TYPES, synth_board, synth_dataset,
                           board_truth_verdict, SEVERITY_RANK)
from steelqg.detector import detect_baseline, detect_improved, decide
from steelqg.evaluate import iou, run_experiment


class TestSynth(unittest.TestCase):
    def test_reproducible(self):
        a, b = synth_board(123, "inclusion"), synth_board(123, "inclusion")
        self.assertTrue(np.array_equal(a.img, b.img))
        self.assertEqual(a.defects[0].bbox, b.defects[0].bbox)

    def test_all_types_have_truth(self):
        for k in DEFECT_TYPES:
            bd = synth_board(1000, k)
            self.assertEqual(len(bd.defects), 1)
            self.assertEqual(bd.defects[0].kind, k)
            self.assertIn(bd.defects[0].severity, SEVERITY_RANK)

    def test_clean_board_has_no_defect(self):
        bd = synth_board(1, "clean")
        self.assertEqual(len(bd.defects), 0)
        self.assertEqual(board_truth_verdict(bd), "OK")

    def test_minor_board_is_ok(self):
        bd = synth_board(1, "scratches")
        self.assertEqual(board_truth_verdict(bd), "OK")


class TestDetector(unittest.TestCase):
    def test_baseline_misses_texture_defects(self):
        # 全局阈值对低对比缺陷（龟裂/点蚀/鳞片/划痕）结构性失效
        recalls = []
        for k in ["crazing", "pitted_surface", "rolled-in_scale", "scratches"]:
            bd = synth_board(2000, k)
            dets = detect_baseline(bd)
            hit = any(iou(d.bbox, bd.defects[0].bbox) > 0.15 for d in dets)
            recalls.append(hit)
        self.assertLess(sum(recalls), 2)

    def test_improved_detects_all(self):
        for k in DEFECT_TYPES:
            bd = synth_board(3000, k)
            dets = detect_improved(bd)
            self.assertTrue(any(iou(d.bbox, bd.defects[0].bbox) > 0.15 for d in dets), k)

    def test_clean_board_no_false_alarm_improved(self):
        bd = synth_board(7, "clean")
        self.assertEqual(len(detect_improved(bd)), 0)

    def test_classification(self):
        expect = {
            "crazing": "crazing", "inclusion": "inclusion", "patches": "patches",
            "pitted_surface": "pitted_surface", "rolled-in_scale": "rolled-in_scale",
            "scratches": "scratches",
        }
        for k, want in expect.items():
            bd = synth_board(1000, k)
            dets = detect_improved(bd)
            self.assertTrue(any(d.kind == want for d in dets), f"{k} -> {[d.kind for d in dets]}")


class TestDecision(unittest.TestCase):
    def test_grade_releases_minor(self):
        bd = synth_board(1000, "scratches")
        dets = detect_improved(bd)
        raw = decide(dets, bd, "raw")
        graded = decide(dets, bd, "grade")
        self.assertEqual(raw["verdict"], "NG")     # 有框即 NG → 过杀
        self.assertEqual(graded["verdict"], "OK")  # 分级放行
        self.assertEqual(graded["truth"], "OK")
        self.assertTrue(graded["correct"])

    def test_rejects_critical(self):
        bd = synth_board(1000, "crazing")
        dets = detect_improved(bd)
        d = decide(dets, bd, "grade")
        self.assertEqual(d["verdict"], "NG")
        self.assertEqual(d["severity"], "critical")

    def test_reasons_explainable(self):
        bd = synth_board(1000, "inclusion")
        d = decide(detect_improved(bd), bd, "grade")
        self.assertGreater(len(d["reasons"]), 0)


class TestIOU(unittest.TestCase):
    def test_identical(self):
        self.assertAlmostEqual(iou((0, 0, 10, 10), (0, 0, 10, 10)), 1.0)

    def test_no_overlap(self):
        self.assertEqual(iou((0, 0, 10, 10), (20, 20, 10, 10)), 0.0)


class TestExperiment(unittest.TestCase):
    def test_three_modes_improve(self):
        r = run_experiment(n_per_type=8, n_clean=8)
        m = {x["mode"]: x for x in r["results"]}
        self.assertLess(m["baseline"]["board_acc"], m["improved"]["board_acc"])
        self.assertLess(m["improved"]["board_acc"], m["improved_grade"]["board_acc"])
        self.assertEqual(m["improved_grade"]["recall"], 1.0)


if __name__ == "__main__":
    unittest.main()
