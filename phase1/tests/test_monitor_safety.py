"""真实服务函数/API 的零成本回归：开放风险、静止复检及长期监测。

python tests/test_monitor_safety.py
不打开摄像头、不启动集群，全部文件在临时目录。
"""
from __future__ import annotations

import copy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fastapi.testclient import TestClient
from PIL import Image
from civility.extract import Window
from civility.prompts import sanitize
from webui import server as S


def risk(severity="high"):
    return {"risk_id": "fall", "name": "人员倒地", "severity": severity,
            "confidence": 0.96, "observable": "人从站立跌倒后躺在地面",
            "alert": True, "alert_reason": "倒地人员可能需要现场帮助"}


def record(i, *, risky=False):
    return {"window_id": f"w{i}", "t_start": i * 2.0,
            "t_end": i * 2.0 + 2, "scene": "走廊", "people": [],
            "violations": {}, "risks": [risk()] if risky else [],
            "raw": "", "error": None, "frames": 1, "frame0": i}


class FakeStream:
    reader = None

    def __init__(self, *args, **kwargs):
        pass

    def ensure(self):
        pass

    def close(self):
        pass


class FakeSession:
    state = "ready"

    def __init__(self):
        self.requests = []

    def infer(self, frames, labels, **kwargs):
        self.requests.append(kwargs)
        # 第二次检查发现名单之外的倒地风险；之后继续观察风险现场。
        parsed = {"scene": "走廊", "violations": [], "people": [],
                  "risks": [risk()] if len(self.requests) >= 2 else []}
        return {"ok": True, "parsed": sanitize(parsed, S.STATE.behaviors),
                "raw": "{}", "error": None, "latency_sec": 0.01}


class MonitorSafetyTests(unittest.TestCase):
    def setUp(self):
        self.saved = (copy.deepcopy(S.STATE.cfg), S.STATE.jobs,
                      S.STATE.uploads, S.STATE.session)
        self.tmp = tempfile.TemporaryDirectory()
        S.STATE.cfg["paths"]["frames_dir"] = self.tmp.name
        S.STATE.jobs = {}
        S.STATE.uploads = {}
        S.STATE.session = None
        self.client = TestClient(S.create_app())

    def tearDown(self):
        self.client.close()
        S.STATE.cfg, S.STATE.jobs, S.STATE.uploads, S.STATE.session = self.saved
        self.tmp.cleanup()

    def run_static_batches(self, batches, session, name):
        """用真实持续任务消费预定批次，保持每批内部的门控与顺序处理。"""
        S.STATE.session = session
        S.STATE.cfg["live"].update(pipeline=False, motion_gate=True)
        frame = Path(self.tmp.name) / f"{name}.jpg"
        Image.new("RGB", (24, 24), "gray").save(frame)
        job = S.JobRecord(name, "static", is_live=True, continuous=True,
                          source_kind="file", source_spec="fake")
        S.STATE.jobs[job.job_id] = job
        pending = iter(batches)

        def capture(*args):
            spans = next(pending, None)
            if spans is None:
                job.stop_flag = True
                return []
            return [Window(f"w{start}", "fake", start, end, [str(frame), str(frame)],
                           [f"t={start}", f"t={end}"], idx0=int(start))
                    for start, end in spans]

        with patch.object(S, "LiveStream", FakeStream), \
                patch.object(S, "capture_live_windows", capture):
            S.run_job(job.job_id)
        self.assertEqual(job.state, "stopped", job.error)
        return job

    def test_open_risks_config_and_prompt_match(self):
        r = self.client.post("/api/aggregate/config", json={"open_risks": True})
        self.assertEqual(r.status_code, 200)
        self.assertIn('"risks"', self.client.get("/api/behaviors/prompt").json()["prompt"])
        self.client.post("/api/aggregate/config", json={"open_risks": False})
        self.assertNotIn('"risks"', self.client.get("/api/behaviors/prompt").json()["prompt"])

    def test_invalid_config_is_atomic(self):
        before = copy.deepcopy(S.STATE.cfg["aggregate"])
        for body in ({"mode": "immediate", "dedup_sec": "NaN"},
                     {"min_confidence": "oops"}, {"open_risks": "false"},
                     {"mode": "immediate", "dedup_sec": "Infinity"}):
            self.assertEqual(self.client.post("/api/aggregate/config", json=body).status_code, 400)
            self.assertEqual(S.STATE.cfg["aggregate"], before)
        before_live = copy.deepcopy(S.STATE.cfg["live"])
        r = self.client.post("/api/live/config", json={"window_stride_sec": 4,
                                                      "motion_recheck_sec": "NaN"})
        self.assertEqual(r.status_code, 400)
        self.assertEqual(S.STATE.cfg["live"], before_live)
        empty = type(S.STATE.behaviors)(behaviors=[], defaults={})
        with patch.object(S.STATE, "behaviors", empty):
            r = self.client.post("/api/aggregate/config", json={"open_risks": False})
            self.assertEqual(r.status_code, 400)
            self.assertEqual(S.STATE.cfg["aggregate"], before)

    def test_gate_rechecks_and_keeps_watching_risks(self):
        job = S.JobRecord("j", "test", continuous=True)
        job.window_results = [record(0)]
        self.assertTrue(S._should_skip_window(job, 0, .5, 10, 30, 2))
        self.assertFalse(S._should_skip_window(job, 0, .5, 32, 30, 2))
        self.assertFalse(S._should_skip_window(job, 2, .5, 10, 30, 2))
        self.assertFalse(S._should_skip_window(job, 0, .5, 10, 30, None))
        job.window_results[-1]["risks"] = [risk()]
        self.assertFalse(S._should_skip_window(job, 0, .5, 10, 30, 2))
        job.window_results[-1] = record(0)
        job.window_results[-1]["error"] = "inference failed"
        self.assertFalse(S._should_skip_window(job, 0, .5, 10, 30, 2))

    def test_continuous_pipeline_static_risk_and_sse(self):
        cfg = S.STATE.cfg
        cfg["aggregate"].update(mode="ai", open_risks=True, dedup_sec=30)
        cfg["live"].update(pipeline=False, motion_gate=True, motion_recheck_sec=6)
        frame = Path(self.tmp.name) / "test.jpg"
        Image.new("RGB", (24, 24), "gray").save(frame)
        session = FakeSession()
        S.STATE.session = session
        job = S.JobRecord("pipeline", "static", is_live=True, continuous=True,
                          source_kind="file", source_spec="fake", with_actions=True)
        S.STATE.jobs[job.job_id] = job
        count = 0

        def capture(*args):
            nonlocal count
            if count == 6:
                job.stop_flag = True
                return []
            i = count
            count += 1
            return [Window(f"w{i}", "fake", i * 2, (i + 1) * 2,
                           [str(frame), str(frame)],
                           [f"t={i * 2}", f"t={(i + 1) * 2}"], idx0=i)]

        with patch.object(S, "LiveStream", FakeStream), patch.object(S, "capture_live_windows", capture):
            S.run_job(job.job_id)
        self.assertEqual(job.state, "stopped", job.error)
        self.assertEqual(len(session.requests), 4)  # first, periodic at 8s, risk followups
        self.assertTrue(all(x["with_risks"] for x in session.requests))
        self.assertGreaterEqual(session.requests[0]["max_new_tokens"], 450 + 384)
        skipped = [w for w in job.window_results if w.get("skipped")]
        self.assertEqual(len(skipped), 2)
        self.assertTrue(all(not w["risks"] and not w["people"] for w in skipped))
        self.assertEqual(len(job.alerts), 1)
        self.assertEqual(job.alerts[0]["source"], "ai_risk")
        events = list(job.events.queue)
        windows = [ev for ev in events if ev["type"] == "window"]
        self.assertTrue(any(ev.get("risks") for ev in windows))
        self.assertEqual(len(windows), 6)
        self.assertEqual(self.client.get("/api/job/pipeline").json()["alerts"], job.alerts)

    def test_history_trim_retains_alerts_and_dedup(self):
        job = S.JobRecord("history", "test", continuous=True, max_kept_windows=3)
        cfg = {"mode": "ai", "min_confidence": .45, "open_risks": True, "dedup_sec": 30}
        for i in range(35):
            job.window_results.append(record(i, risky=True))
            job.window_results = job.window_results[-job.max_kept_windows:]
            S._refresh_alerts(job, "test", cfg)
        self.assertEqual(len(job.alerts), 3, job.alerts)
        starts = [a["t_start"] for a in job.alerts]
        self.assertEqual(starts, [0, 30, 60])
        for i in range(35, 40):
            job.window_results = [record(i)]
            S._refresh_alerts(job, "test", cfg)
        self.assertEqual(len(job.alerts), 3, "旧告警应保留")

    def test_same_batch_risk_keeps_following_static_windows_under_observation(self):
        S.STATE.cfg["aggregate"].update(mode="ai", open_risks=True, dedup_sec=30)
        S.STATE.cfg["live"]["motion_recheck_sec"] = 30
        session = FakeSession()
        job = self.run_static_batches([[(0, 2)], [(30, 32), (32, 34)], [(34, 36)]],
                                      session, "batch-risk")
        self.assertEqual(len(session.requests), 4,
                         "同批发现风险后，原定静止跳过窗口和后批都必须继续检查")
        self.assertTrue(all(not r.get("skipped") for r in job.window_results))
        self.assertTrue(all(r["risks"] for r in job.window_results[1:]))
        self.assertEqual(len(job.alerts), 1, "风险跟进不应突破同类去重")

    def test_same_batch_failed_inference_cannot_turn_next_window_into_normal(self):
        S.STATE.cfg["aggregate"].update(mode="ai", open_risks=True, dedup_sec=30)
        S.STATE.cfg["live"]["motion_recheck_sec"] = 30

        class FailingSession(FakeSession):
            def infer(self, frames, labels, **kwargs):
                response = super().infer(frames, labels, **kwargs)
                if len(self.requests) == 2:
                    return {"ok": False, "parsed": None, "error": "temporary failure",
                            "latency_sec": .01}
                return response

        session = FailingSession()
        job = self.run_static_batches([[(0, 2)], [(30, 32), (32, 34)], [(34, 36)]],
                                      session, "batch-error")
        self.assertEqual(len(session.requests), 4)
        self.assertEqual(job.window_results[1]["error"], "temporary failure")
        self.assertFalse(job.window_results[2].get("skipped"),
                         "失败之后的同批窗口必须重新检查，不能发布空白正常结论")
        self.assertTrue(job.window_results[2]["risks"])
        self.assertEqual(len(job.alerts), 1)

    def test_live_risk_and_recheck_settings_take_effect_on_next_batch(self):
        S.STATE.cfg["aggregate"].update(mode="immediate", open_risks=False, dedup_sec=30)
        S.STATE.cfg["live"]["motion_recheck_sec"] = 30
        client = self.client
        testcase = self

        class UpdatingSession(FakeSession):
            def infer(self, frames, labels, **kwargs):
                response = super().infer(frames, labels, **kwargs)
                if len(self.requests) == 1:
                    testcase.assertEqual(client.post("/api/aggregate/config", json={
                        "mode": "ai", "open_risks": True, "dedup_sec": 0,
                    }).status_code, 200)
                    testcase.assertEqual(client.post("/api/live/config", json={
                        "motion_recheck_sec": 1,
                    }).status_code, 200)
                return response

        session = UpdatingSession()
        job = self.run_static_batches([[(0, 2)], [(2, 4)], [(4, 6)]],
                                      session, "hot-settings")
        self.assertEqual(len(session.requests), 3,
                         "缩短静止复检间隔应在下一批实际送模型")
        self.assertFalse(session.requests[0]["with_risks"])
        self.assertFalse(session.requests[0]["with_alert"])
        self.assertTrue(all(r["with_risks"] and r["with_alert"]
                            for r in session.requests[1:]))
        self.assertGreater(session.requests[1]["max_new_tokens"],
                           session.requests[0]["max_new_tokens"])
        self.assertEqual(len(job.alerts), 2,
                         "去重改为零后，应按最新配置分别发出新风险提醒")

    def test_zero_frame_and_global_frame_after_trim(self):
        job = S.JobRecord("frames", "test")
        S.STATE.jobs[job.job_id] = job
        frames_dir = Path(self.tmp.name) / job.job_id
        frames_dir.mkdir()
        (frames_dir / "f_00000.jpg").write_bytes(b"frame-zero")
        (frames_dir / "f_00100.jpg").write_bytes(b"frame-hundred")
        job.window_results = [dict(record(0), t_start=90, frame0=0),
                              dict(record(1), frame0=100), dict(record(2), frame0=50)]
        r = self.client.get("/api/job/frames/frames/w0/0")
        self.assertEqual(r.content, b"frame-zero")
        r = self.client.get("/api/job/frames/frames/w1/0")
        self.assertEqual(r.content, b"frame-hundred")
        self.assertEqual(self.client.get("/api/job/frames/frames/w2/0").status_code, 404)
        self.assertEqual(self.client.get("/api/job/frames/frames/w1/1").status_code, 404)

    def test_recoverable_sse_error_does_not_close_stream(self):
        job = S.JobRecord("sse", "test", state="done")
        S.STATE.jobs[job.job_id] = job
        job.events.put({"type": "error", "error": "reconnecting", "recovering": True})
        job.events.put({"type": "done", "summary": job.summary()})
        r = self.client.get("/api/job/sse/stream")
        self.assertIn('"recovering": true', r.text)
        self.assertIn('"type": "done"', r.text)

    def test_missing_service_reports_original_error(self):
        job = S.JobRecord("no-service", "test")
        S.STATE.jobs[job.job_id] = job
        S.run_job(job.job_id)
        self.assertEqual(job.state, "error")
        self.assertIn("推理服务未就绪", job.error)

    def test_old_worker_or_invalid_schema_cannot_look_successful(self):
        frame = Path(self.tmp.name) / "input.jpg"
        frame.write_bytes(b"test frame")
        S.STATE.cfg["aggregate"]["open_risks"] = True
        for result in (
            {"parsed": {"scene": "走廊", "violations": {}}, "error": None},
            {"parsed": {"scene": "走廊", "violations": {}, "risks": []},
             "valid": False, "problems": ["invalid risk"], "error": None},
        ):
            with self.subTest(result=result):
                class Session:
                    state = "ready"

                    def infer(self, *args, **kwargs):
                        return dict(result)

                S.STATE.session = Session()
                job = S.JobRecord("invalid-worker", "test", cached_windows=[
                    Window("w0", "fake", 0, 2, [str(frame)], ["t=0"], idx0=0)])
                S.STATE.jobs[job.job_id] = job
                S.STATE.uploads[job.job_id] = frame
                S.run_job(job.job_id)
                self.assertTrue(job.window_results[0]["error"])
                self.assertEqual(job.window_results[0]["risks"], [])
                self.assertIsNone(job.last_model_t_end)
                self.assertEqual(job.alerts, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
