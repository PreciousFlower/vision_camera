"""Protect short live actions when inference skips ahead to recent windows."""
import copy
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from webui import server as S
from webui.motion_highlights import MotionHighlights, action_frames, merge_sampling


class GapActionTests(unittest.TestCase):
    def test_sixteen_second_inference_keeps_gap_action_and_four_current_frames(self):
        cfg = copy.deepcopy(S.STATE.cfg)
        cfg['live'].update(window_stride_sec=2, continuous_batch_windows=2,
                           frames_per_window=4, context_sec=4)
        cfg['extract']['max_frames_per_window'] = 18
        history = [(100 + i / 8, f'jpeg-{i}'.encode()) for i in range(193)]
        reader = types.SimpleNamespace(dead=False, snapshot_history=lambda: history,
            snapshot_highlights=lambda: [(110.0, 20.0)])
        job = S.JobRecord('gap-action', 'camera', is_live=True, continuous=True,
            source_kind='local', capture_started_at=100, last_recent_wall=108,
            last_batch_sec=16)
        with tempfile.TemporaryDirectory() as directory:
            cfg['paths']['frames_dir'] = directory
            with patch.object(S.STATE, 'cfg', cfg), \
                    patch.object(S.time, 'time', return_value=124.1), \
                    patch.object(S.STATE, 'log_line'):
                windows = S.capture_recent_windows(job, reader)
        self.assertIn('t=10.0s', windows[0].labels,
                      'A release lasting only 125ms during inference must reach the model')
        first_current = [float(t[2:-1]) for t in windows[0].labels
                         if 20 < float(t[2:-1]) <= 22]
        self.assertEqual(len(first_current), 4)
        self.assertEqual([w.t_end for w in windows], [22, 24])
        self.assertTrue(all(len(w.frames) <= 12 for w in windows),
                        'Motion preservation must not increase the normal image budget')
        self.assertTrue(all(float(label[2:-1]) > 16 for label in windows[1].labels))
        self.assertEqual(cfg['live']['window_stride_sec'], 2)
        self.assertEqual(windows[0].sampling['retained_action_frames'], 3)
        self.assertEqual(windows[0].sampling['gap_sec'], 8)
        self.assertEqual(windows[0].sampling['unexamined_gap_sec'], 0)
        self.assertEqual(windows[0].sampling['frame_times'][1], 10)

    def test_later_release_is_not_displaced_by_two_stronger_walking_peaks(self):
        history = [(i / 8, b'frame') for i in range(80)]
        highlights = [(2.25, 56.89), (3.375, 50.47), (6.625, 48.84)]
        chosen = action_frames(history, highlights, start=0, end=9, max_frames=6)
        self.assertEqual([t for t, _ in chosen], [1.875, 2.25, 2.625, 6.25, 6.625, 7.0])

    def test_expired_buffer_gap_is_reported_without_unbounded_recovery(self):
        cfg = copy.deepcopy(S.STATE.cfg)
        cfg['live'].update(window_stride_sec=2, continuous_batch_windows=2,
                           frames_per_window=4, context_sec=4)
        history = [(120 + i / 8, b'frame') for i in range(241)]
        reader = types.SimpleNamespace(dead=False, snapshot_history=lambda: history,
            snapshot_highlights=lambda: [(122, 10)])
        job = S.JobRecord('expired', 'camera', is_live=True, continuous=True,
            source_kind='local', capture_started_at=100, last_recent_wall=108)
        with tempfile.TemporaryDirectory() as directory:
            cfg['paths']['frames_dir'] = directory
            with patch.object(S.STATE, 'cfg', cfg), \
                    patch.object(S.time, 'time', return_value=150.1), \
                    patch.object(S.STATE, 'log_line'):
                windows = S.capture_recent_windows(job, reader)
        self.assertEqual(len(windows), 2)
        self.assertEqual(job.live_metrics['sampled_gap_sec'], 34)
        self.assertEqual(job.live_metrics['unexamined_gap_sec'], 12)
        self.assertLessEqual(len(windows[0].frames), 12)

    def test_exposure_changes_do_not_displace_small_local_actions(self):
        import numpy as np
        motion = MotionHighlights(limit=8)
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        motion.add(0, frame)
        motion.add(1, frame + 30)
        self.assertEqual(motion.scores[-1][1], 0)
        changed = frame + 30
        changed[16:24, 24:32] = 120
        motion.add(2, changed)
        self.assertGreater(motion.scores[-1][1], 80)
        for i in range(100):
            motion.add(i + 3, changed)
        self.assertEqual(len(motion.scores), 8)
        self.assertEqual(motion.previous.size, 64 * 48)
        motion.clear()
        self.assertFalse(motion.scores)
        self.assertIsNone(motion.previous)

    def test_sampling_remains_unique_sorted_and_preserves_all_new_frames(self):
        prior = [(float(i), b'old') for i in range(8)]
        fresh = [(float(i), b'new') for i in range(8, 12)]
        peaks = [(-3, b'peak'), (-2, b'peak'), (-1, b'peak')]
        selected = merge_sampling(prior, fresh, peaks, budget=12)
        times = [t for t, _ in selected]
        self.assertEqual(times, sorted(set(times)))
        self.assertTrue(set(range(8, 12)).issubset(times))
        self.assertTrue({0, 7}.issubset(times))
        self.assertEqual(len(selected), 12)
        all_peaks = [(float(i), b'peak') for i in range(-6, 0)]
        selected = merge_sampling(prior, fresh, all_peaks, budget=12)
        self.assertEqual(len(selected), 12)
        self.assertTrue(set(range(-6, 0)).issubset(t for t, _ in selected))

    def test_retained_motion_cannot_be_skipped_by_low_resolution_motion_gate(self):
        cfg = copy.deepcopy(S.STATE.cfg)
        cfg['live'].update(window_stride_sec=2, continuous_batch_windows=2,
                           frames_per_window=4, context_sec=4, motion_gate=True)
        history = [(100 + i / 8, b'frame') for i in range(193)]
        reader = types.SimpleNamespace(dead=False, snapshot_history=lambda: history,
            snapshot_highlights=lambda: [(110, 20)])
        stream = types.SimpleNamespace(reader=reader, kind='local',
            ensure=lambda: None, ensure_reader=lambda **kw: None, close=lambda: None)
        job = S.JobRecord('gate', 'camera', is_live=True, continuous=True,
            source_kind='local', source_spec='0', capture_started_at=100,
            last_recent_wall=108, last_batch_sec=16, last_model_t_end=8)
        job.window_results = [{'window_id': 'old', 't_start': 6, 't_end': 8,
            'violations': {}, 'scene': 'normal', 'raw': '{}', 'error': None}]
        submitted = []
        def infer(frames, labels, **kw):
            submitted.append(labels)
            job.stop_flag = True
            return {'ok': True, 'parsed': {'scene': '正常', 'people': [],
                'violations': {}, 'risks': []}, 'error': None, 'raw': '{}'}
        with tempfile.TemporaryDirectory() as directory:
            cfg['paths']['frames_dir'] = directory
            with patch.object(S.STATE, 'cfg', cfg), patch.object(S.STATE, 'jobs', {'gate': job}), \
                    patch.object(S.STATE, 'session', types.SimpleNamespace(state='ready', infer=infer)), \
                    patch.object(S, 'LiveStream', return_value=stream), \
                    patch.object(S.time, 'time', return_value=124.1), \
                    patch('webui.capture.frame_motion', return_value=0), \
                    patch.object(S.STATE, 'log_line'):
                S.run_job('gate')
        self.assertEqual(len(submitted), 1)
        self.assertIn('t=10.0s', submitted[0])
        self.assertEqual(job.state, 'stopped', job.error)

    def test_alert_age_uses_old_cited_evidence_not_recent_window_tail(self):
        job = S.JobRecord('timing', 'camera', is_live=True, continuous=True,
                         source_kind='local', capture_started_at=100)
        job.window_results = [{'window_id': 'w', 't_end': 24,
            'frame_times': [10, 10.375, 22, 24],
            'violations': {'littering': {'evidence': {'release': 1, 'landing': 2}}}}]
        alert = types.SimpleNamespace(window_ids=['w'], behavior_id='littering', t_end=24)
        with patch.object(S.time, 'time', return_value=130):
            timing = S._live_alert_timing(job, alert)
        self.assertEqual(timing['notification_age_sec'], 19.625)
        self.assertEqual(timing['window_age_sec'], 6)
        self.assertEqual(timing['evidence_time_sec'], 10.375)
        alert.behavior_id = 'ai_risk:fall'
        with patch.object(S.time, 'time', return_value=130):
            timing = S._live_alert_timing(job, alert)
        self.assertIsNone(timing['evidence_age_sec'])
        self.assertEqual(timing['window_age_sec'], 6)

    def test_quiet_motion_gate_does_not_freeze_last_decision_age(self):
        job = S.JobRecord('quiet', 'camera', state='running', is_live=True,
                         source_kind='local', last_model_frame_at=120, result_age_sec=6)
        with patch.object(S.time, 'time', return_value=140):
            self.assertEqual(job.summary()['result_age_sec'], 20)

    def test_new_smoking_alert_after_dedup_uses_evidence_time_not_old_context(self):
        job = S.JobRecord('dedup', 'camera', is_live=True, continuous=True,
                         source_kind='local', capture_started_at=100)
        def smoking_record(wid, start, end, times, evidence_ref):
            return {'window_id': wid, 't_start': start, 't_end': end,
                'frame_times': times, 'sampling': {'mode': 'recent_with_motion'},
                'scene': '有人吸烟', 'people': [], 'risks': [], 'raw': '{}', 'error': None,
                'violations': {'smoking': {'violation': True, 'confidence': .95,
                    'observable': '可辨认烟支在嘴边吸食', 'alert': True,
                    'alert_reason': '明确吸烟', 'evidence': {'tobacco': evidence_ref, 'use': evidence_ref}}}}
        cfg = {**S.STATE.cfg['aggregate'], 'mode': 'ai', 'dedup_sec': 30}
        job.window_results = [smoking_record('old', 0, 12, [0, 10, 12], 2)]
        with patch.object(S.time, 'time', return_value=113):
            S._refresh_alerts(job, 'camera', cfg)
        self.assertEqual(len(job.alerts), 1)
        job.window_results.append(smoking_record('new', 24.625, 46,
            [24.625, 25, 25.375, 40, 42, 45, 46], 6))
        with patch.object(S.time, 'time', return_value=148):
            S._refresh_alerts(job, 'camera', cfg)
        self.assertEqual(len(job.alerts), 2,
                         'Old unrelated motion must not suppress smoking 35s after prior evidence')
        self.assertEqual(job.alerts[-1]['t_start'], 24.625)
        self.assertEqual(job.alerts[-1]['evidence_time_sec'], 45)
        self.assertEqual(job.live_metrics['last_alert_age_sec'], 3)

    def test_suppressed_alert_does_not_replace_last_notification_metrics(self):
        job = S.JobRecord('suppressed', 'camera', is_live=True, continuous=True,
                         source_kind='local', capture_started_at=100)
        record = {'window_id': 'w', 't_start': 10, 't_end': 12,
            'frame_times': [10, 12], 'sampling': {'mode': 'recent'},
            'scene': '有人吸烟', 'people': [], 'risks': [], 'raw': '{}', 'error': None,
            'violations': {'smoking': {'violation': True, 'confidence': .95,
                'observable': '可辨认烟支在嘴边吸食', 'alert': True,
                'alert_reason': '明确吸烟', 'evidence': {'tobacco': 1, 'use': 1}}}}
        cfg = {**S.STATE.cfg['aggregate'], 'mode': 'ai', 'dedup_sec': 30}
        job.window_results = [record]
        with patch.object(S.time, 'time', return_value=113):
            S._refresh_alerts(job, 'camera', cfg)
        saved = dict(job.live_metrics)
        job.window_results.append({**record, 'window_id': 'next', 't_end': 14,
            'frame_times': [11, 14]})
        with patch.object(S.time, 'time', return_value=119):
            S._refresh_alerts(job, 'camera', cfg)
        self.assertEqual(len(job.alerts), 1)
        self.assertEqual(job.live_metrics, saved)

    def test_legacy_live_alert_keeps_existing_window_start_dedup(self):
        job = S.JobRecord('legacy', 'camera', is_live=True, continuous=True,
                         source_kind='local', capture_started_at=100)
        record = {'window_id': 'w', 't_start': 0, 't_end': 12,
            'frame_times': [10, 12], 'scene': '有人吸烟', 'people': [], 'risks': [],
            'raw': '{}', 'error': None,
            'violations': {'smoking': {'violation': True, 'confidence': .95,
                'observable': '可辨认烟支在嘴边吸食', 'alert': True,
                'alert_reason': '明确吸烟', 'evidence': {'tobacco': 1, 'use': 1}}}}
        cfg = {**S.STATE.cfg['aggregate'], 'mode': 'ai', 'dedup_sec': 30}
        job.window_results = [record]
        with patch.object(S.time, 'time', return_value=113):
            S._refresh_alerts(job, 'camera', cfg)
        self.assertEqual(job.alert_last_by_type['smoking'][0], 0)
        job.window_results.append({**record, 'window_id': 'next', 't_start': 24.625,
            't_end': 46, 'frame_times': [45, 46]})
        with patch.object(S.time, 'time', return_value=148):
            S._refresh_alerts(job, 'camera', cfg)
        self.assertEqual(len(job.alerts), 1)

    def test_live_risk_without_numbered_evidence_dedups_by_window_end(self):
        job = S.JobRecord('risk-dedup', 'camera', is_live=True, continuous=True,
                         source_kind='local', capture_started_at=100)
        record = {'window_id': 'old', 't_start': 0, 't_end': 12,
            'frame_times': [10, 12], 'sampling': {'mode': 'recent'},
            'scene': '有明火', 'people': [], 'raw': '{}', 'error': None, 'violations': {},
            'risks': [{'risk_id': 'fire', 'name': '明火', 'severity': 'high', 'confidence': .95,
                'observable': '电线旁有可见明火', 'alert': True, 'alert_reason': '存在火灾风险'}]}
        cfg = {**S.STATE.cfg['aggregate'], 'mode': 'ai', 'dedup_sec': 30, 'open_risks': True}
        job.window_results = [record]
        with patch.object(S.time, 'time', return_value=113):
            S._refresh_alerts(job, 'camera', cfg)
        self.assertEqual(job.alert_last_by_type['ai_risk:fire'][0], 12)
        job.window_results.append({**record, 'window_id': 'new', 't_start': 24.625,
            't_end': 46, 'frame_times': [24.625, 45, 46]})
        with patch.object(S.time, 'time', return_value=148):
            S._refresh_alerts(job, 'camera', cfg)
        self.assertEqual(len(job.alerts), 2)
        self.assertIsNone(job.alerts[-1]['evidence_time_sec'])
        self.assertIsNone(job.alerts[-1]['evidence_age_sec'])
        self.assertEqual(job.alerts[-1]['window_age_sec'], 2)

    def test_twenty_or_forty_second_stride_is_reported_as_cache_configuration(self):
        cfg = copy.deepcopy(S.STATE.cfg)
        cfg['live'].update(continuous_batch_windows=2, context_sec=4)
        reader = types.SimpleNamespace(dead=False,
            snapshot_history=lambda: [(8.125 + i / 8, b'frame') for i in range(256)])
        for stride in (20, 40):
            cfg['live']['window_stride_sec'] = stride
            job = S.JobRecord('cache-config', 'camera', is_live=True, continuous=True,
                source_kind='local', capture_started_at=0, last_recent_wall=0)
            with self.subTest(stride=stride), patch.object(S.STATE, 'cfg', cfg), \
                    patch.object(S.time, 'time', return_value=40.1):
                with self.assertRaisesRegex(ValueError, '配置.*缓存'):
                    S.capture_recent_windows(job, reader)

    def test_missing_normal_context_is_included_in_unexamined_timeline(self):
        cfg = copy.deepcopy(S.STATE.cfg)
        cfg['live'].update(window_stride_sec=2, continuous_batch_windows=2, context_sec=4)
        history = [(124 + i / 8, b'frame') for i in range(33)]
        reader = types.SimpleNamespace(dead=False, snapshot_history=lambda: history)
        job = S.JobRecord('truncated', 'camera', is_live=True, continuous=True,
            source_kind='local', capture_started_at=100, last_recent_wall=124)
        with tempfile.TemporaryDirectory() as directory:
            cfg['paths']['frames_dir'] = directory
            with patch.object(S.STATE, 'cfg', cfg), \
                    patch.object(S.time, 'time', return_value=128.1), \
                    patch.object(S.STATE, 'log_line'):
                S.capture_recent_windows(job, reader)
        self.assertEqual(job.live_metrics['sampled_gap_sec'], 0)
        self.assertEqual(job.live_metrics['unexamined_gap_sec'], 4)

    def test_oversized_sampling_stops_job_without_reconnecting_camera(self):
        cfg = copy.deepcopy(S.STATE.cfg)
        cfg['live'].update(window_stride_sec=40, continuous_batch_windows=2, context_sec=4)
        reader = types.SimpleNamespace(dead=False, snapshot_history=lambda: [])
        closes = []
        reconnects = []
        stream = types.SimpleNamespace(reader=reader, kind='local', ensure=lambda: None,
            ensure_reader=lambda **kw: None, close=lambda: closes.append(True),
            drop=lambda reason: reconnects.append(reason))
        job = S.JobRecord('oversized', 'camera', is_live=True, continuous=True,
            source_kind='local', source_spec='0')
        with patch.object(S.STATE, 'cfg', cfg), patch.object(S.STATE, 'jobs', {'oversized': job}), \
                patch.object(S.STATE, 'session', types.SimpleNamespace(state='ready')), \
                patch.object(S, 'LiveStream', return_value=stream), patch.object(S.STATE, 'log_line'):
            S.run_job('oversized')
        self.assertEqual(job.state, 'error')
        self.assertIn('配置', job.error)
        self.assertIn('缓存', job.error)
        self.assertEqual(reconnects, [])
        self.assertEqual(closes, [True])


if __name__ == '__main__':
    unittest.main(verbosity=2)
