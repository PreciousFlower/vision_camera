#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""开放风险评分：未标注告警单独统计，明确标注的风险才参与准确率。"""
from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from civility.aggregate import Alert
from civility.config import Behavior, BehaviorSet
from civility.score import Label, render_report, score_events, score_windows
from civility.vlm import WindowResult


BS = BehaviorSet([Behavior("smoking", "吸烟", "可见吸烟动作", severity="high")], {})
EMPTY = BehaviorSet([], {})


def alert(bid="ai_risk:fall", video="camera.mp4", start=0, end=2, source="ai_risk") -> Alert:
    return Alert(behavior_id=bid, behavior_name="人员摔倒" if bid.startswith("ai_risk:") else "吸烟",
                 severity="high", video=video, t_start=start, t_end=end, hits=1, span=1,
                 max_confidence=0.95, evidences=["可见动作证据"], source=source)


class RiskScoringTests(unittest.TestCase):
    def test_unlabeled_risks_preserve_rule_metrics(self):
        alerts = [alert("smoking", source="configured"), alert(), alert("ai_risk:fire")]
        labels = [Label("camera.mp4", "smoking", 0, 2)]
        score = score_events(alerts, labels, BS)
        self.assertEqual(score["alerts_total"], 3)
        self.assertEqual(score["alerts_scored_total"], 1)
        self.assertEqual(score["unscored_alerts_total"], 2)
        self.assertEqual(score["open_risk_alerts_total"], 2)
        self.assertEqual(score["open_risk_alerts_scored_total"], 0)
        self.assertEqual(score["open_risk_alerts_unscored_total"], 2)
        self.assertEqual(score["overall"], score["configured_overall"])
        self.assertEqual(score["overall"]["tp"], 1)
        self.assertEqual(score["overall"]["fp"], 0)
        self.assertEqual(score["overall"]["precision"], 1)
        self.assertNotIn("ai_risk:fall", score["per_behavior"])
        self.assertIsNone(score["open_risk_overall"]["precision"])
        self.assertEqual(score["unscored_alerts_by_behavior"], {"ai_risk:fall": 1, "ai_risk:fire": 1})

    def test_annotated_risks_have_tp_fp_fn_and_separate_rule_metrics(self):
        alerts = [alert("smoking", source="configured"), alert(),
                  alert(start=10, end=12), alert("ai_risk:conflict")]
        labels = [Label("camera.mp4", "smoking", 0, 2),
                  Label("camera.mp4", "ai_risk:fall", 0, 2),
                  Label("camera.mp4", "ai_risk:fire", 20, 22)]
        score = score_events(alerts, labels, BS)
        self.assertEqual(score["alerts_scored_total"], 3)
        self.assertEqual(score["unscored_alerts_total"], 1)
        self.assertEqual(score["open_risk_alerts_total"], 3)
        self.assertEqual(score["open_risk_alerts_scored_total"], 2)
        self.assertEqual(score["configured_overall"]["tp"], 1)
        self.assertEqual(score["open_risk_overall"]["tp"], 1)
        self.assertEqual(score["open_risk_overall"]["fp"], 1)
        self.assertEqual(score["open_risk_overall"]["fn"], 1)
        self.assertEqual(score["overall"]["tp"], 2)
        self.assertEqual(score["per_behavior"]["ai_risk:fire"]["fn"], 1)

    def test_risk_labels_apply_only_to_annotated_video_and_category(self):
        alerts = [alert(), alert(video="unlabeled.mp4"), alert("ai_risk:fire")]
        score = score_events(alerts, [Label("camera.mp4", "ai_risk:fall", 0, 2)], EMPTY)
        self.assertEqual(score["alerts_scored_total"], 1)
        self.assertEqual(score["unscored_alerts_total"], 2)
        self.assertEqual(score["overall"]["fp"], 0)

    def test_duplicate_alert_cannot_reuse_label_to_inflate_recall(self):
        score = score_events([alert(), alert()], [Label("camera.mp4", "ai_risk:fall", 0, 2)], EMPTY)
        self.assertEqual(score["open_risk_overall"]["tp"], 1)
        self.assertEqual(score["open_risk_overall"]["fp"], 1)
        self.assertEqual(score["open_risk_overall"]["recall"], 1)

    def test_empty_behavior_set_risk_only_is_safe_with_and_without_labels(self):
        unlabeled = score_events([alert()], [], EMPTY)
        self.assertEqual(unlabeled["alerts_total"], 1)
        self.assertEqual(unlabeled["alerts_scored_total"], 0)
        self.assertEqual(unlabeled["unscored_alerts_total"], 1)
        self.assertEqual(unlabeled["per_behavior"], {})
        for metric in ("precision", "recall", "f1"):
            self.assertIsNone(unlabeled["overall"][metric])
        labeled = score_events([alert()], [Label("camera.mp4", "ai_risk:fall", 0, 2)], EMPTY)
        self.assertEqual(labeled["alerts_scored_total"], 1)
        self.assertEqual(labeled["overall"]["tp"], 1)
        self.assertEqual(labeled["configured_overall"]["tp"], 0)

    def test_old_alert_source_and_unknown_behavior_are_safe(self):
        legacy_risk = alert(source="configured")
        score = score_events([legacy_risk, alert("disabled_rule", source="configured")], [], BS)
        self.assertEqual(score["unscored_alerts_total"], 2)
        self.assertEqual(score["open_risk_alerts_total"], 1)
        self.assertEqual(score["overall"]["fp"], 0)

    def test_report_explains_unscored_risks_and_window_metric_scope(self):
        window = score_windows([], [], EMPTY)
        event = score_events([alert()], [], EMPTY)
        report = render_report(window, event, EMPTY)
        self.assertIn("告警总数：1 ／ 已评分：0 ／ 未评分：1", report)
        self.assertIn("开放风险告警：1 ／ 已评分：0 ／ 未评分：1", report)
        self.assertIn("不计为 TP 或 FP", report)
        self.assertIn("不能据此报告判断风险准确率", report)
        self.assertIn("窗口级指标不代表开放风险识别能力", report)

    def test_report_with_annotations_displays_risk_metrics(self):
        window = score_windows([], [], EMPTY)
        event = score_events([alert()], [Label("camera.mp4", "ai_risk:fall", 0, 2)], EMPTY)
        report = render_report(window, event, EMPTY)
        self.assertIn("ai_risk:fall", report)
        self.assertIn("开放风险告警：1 ／ 已评分：1 ／ 未评分：0", report)
        self.assertNotIn("尚无开放风险标注", report)

    def test_cli_score_writes_report_when_open_risk_has_no_annotation(self):
        from civility.cli import cmd_score

        with tempfile.TemporaryDirectory(prefix="civility-risk-score-") as temp:
            root = Path(temp)
            result = WindowResult("w0", "camera.mp4", 0, 2,
                                  parsed={"scene": "门口有人摔倒", "violations": [{"behavior_id": "smoking",
                                                                                   "violation": True,
                                                                                   "confidence": 0.9,
                                                                                   "observable": "另一人举烟至嘴边",
                                                                                   "alert": True}],
                                          "risks": [{"risk_id": "fall", "name": "人员摔倒", "severity": "high",
                                                     "confidence": 0.95, "observable": "中央人物由站立倒地",
                                                     "alert": True, "alert_reason": "需要现场确认"}]})
            input_path = root / "windows.jsonl"
            input_path.write_text(json.dumps(asdict(result), ensure_ascii=False) + "\n", encoding="utf-8")
            annotations = root / "annotations.jsonl"
            annotations.write_text(json.dumps({"video": "camera.mp4", "behavior_id": "smoking",
                                               "start": 0, "end": 2}) + "\n", encoding="utf-8")
            report_path = root / "report.md"
            args = SimpleNamespace(manifest=str(root / "manifest.jsonl"), input=str(input_path),
                                   annotations=str(annotations), out=str(report_path))
            cfg = {"aggregate": {"mode": "ai", "open_risks": True, "dedup_sec": 30},
                   "paths": {"results_dir": str(root / "results"), "data_dir": str(root)}}
            with patch("civility.cli._cfg", return_value=(cfg, BS)), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(cmd_score(args), 0)
            report = json.loads(report_path.with_suffix(".json").read_text(encoding="utf-8"))
            self.assertEqual(report["event"]["alerts_total"], 2)
            self.assertEqual(report["event"]["alerts_scored_total"], 1)
            self.assertEqual(report["event"]["unscored_alerts_total"], 1)
            self.assertEqual(report["event"]["overall"]["tp"], 1)
            self.assertEqual(report["event"]["overall"]["fp"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
