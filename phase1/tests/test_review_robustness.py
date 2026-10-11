"""Regression cases found during the robustness review; no GPU or camera."""
import copy
from dataclasses import asdict
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from civility import runner
from civility.config import load_behaviors
from civility.extract import Window
from civility.vlm import WindowResult
from webui import server as S


class WatchdogRobustnessTests(unittest.TestCase):
    def setUp(self):
        self.saved = (copy.deepcopy(S.STATE.cfg), S.STATE.jobs,
                      S.STATE.session, S.STATE.cap_extended_at)
        S.STATE.cfg['service'] = {'idle_release_min': 0, 'max_run_hours': 1}
        S.STATE.session = types.SimpleNamespace(state='ready', started_at=0)
        S.STATE.cap_extended_at = 0

    def tearDown(self):
        S.STATE.cfg, S.STATE.jobs, S.STATE.session, S.STATE.cap_extended_at = self.saved

    def test_active_job_survives_repeated_ticks_during_extension(self):
        S.STATE.session.started_at = 100
        S.STATE.jobs = {'live': types.SimpleNamespace(state='running')}
        with patch.object(S.time, 'time', return_value=7300) as clock, \
                patch.object(S, '_auto_release') as release, \
                patch.object(S.STATE, 'log_line') as log:
            S.watchdog_tick()
            for now in (7310, 7400, 9101, 9111):
                clock.return_value = now
                S.watchdog_tick()
            release.assert_not_called()
            self.assertEqual(log.call_count, 2)
            S.STATE.jobs['live'].state = 'done'
            S.watchdog_tick()
            release.assert_called_once()

    def test_extracting_video_is_active_work(self):
        S.STATE.jobs = {'file': types.SimpleNamespace(state='extracting')}
        self.assertTrue(S._active_job())

    def test_failed_automatic_release_does_not_claim_node_was_freed(self):
        with patch.object(S.STATE, 'session', types.SimpleNamespace(
                state='ready', started_at=100,
                stop=Mock(side_effect=RuntimeError('release offline')))), \
                patch.object(S.STATE, 'auto_released', ''), \
                patch.object(S.STATE, 'log_line') as log:
            message = S._auto_release('超过最长运行上限')
            self.assertIn('自动释放失败', message)
            self.assertIn('release offline', message)
            self.assertNotIn('已释放节点', message)
            self.assertNotIn('省下', message)
            self.assertEqual(S.STATE.auto_released, message)
            self.assertIn('自动释放失败', log.call_args.args[0])


class CheckpointRobustnessTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / 'results.jsonl'
        self.bs = load_behaviors()
        self.windows = [Window('w'+str(i), 'fixture.mp4', i*2, i*2+2, [], [])
                        for i in range(2)]

    def tearDown(self):
        self.tmp.cleanup()

    @staticmethod
    def result(request):
        return WindowResult(request.window_id, request.video, request.t_start,
                            request.t_end, {'scene': 'normal', 'violations': {}, 'people': []})

    def backend(self):
        return types.SimpleNamespace(name='fixture', infer=lambda req, bs:self.result(req))

    def test_resume_after_interrupted_tail_keeps_new_result_readable(self):
        first = asdict(self.result(self.windows[0].to_request()))
        self.path.write_text(json.dumps(first)+'\n'+ '{"window_id":"interrupted', encoding='utf-8')
        stats = runner.run_windows(self.windows, self.backend(), self.bs, self.path, verbose=False)
        self.assertEqual(stats['ran_now'], 1)
        self.assertEqual([r.window_id for r in runner.load_results(self.path)], ['w0', 'w1'])
        self.assertEqual(set(runner.load_done(self.path)), {'w0', 'w1'})

    def test_valid_last_line_without_newline_is_not_concatenated(self):
        first = asdict(self.result(self.windows[0].to_request()))
        self.path.write_text(json.dumps(first), encoding='utf-8')
        runner.run_windows(self.windows, self.backend(), self.bs, self.path, verbose=False)
        self.assertEqual([r.window_id for r in runner.load_results(self.path)], ['w0', 'w1'])

    def test_non_object_checkpoint_lines_do_not_break_valid_results(self):
        first = asdict(self.result(self.windows[0].to_request()))
        self.path.write_text('null\n[]\n23\n'+json.dumps(first)+'\n', encoding='utf-8')
        self.assertEqual(set(runner.load_done(self.path)), {'w0'})
        self.assertEqual([r.window_id for r in runner.load_results(self.path)], ['w0'])

    def test_parallel_executor_is_closed_after_success_and_error(self):
        original = runner.ThreadPoolExecutor
        executors = []
        def create(**kwargs):
            executor = original(**kwargs)
            executors.append(executor)
            return executor
        with patch.object(runner, 'ThreadPoolExecutor', side_effect=create):
            runner.run_windows(self.windows, self.backend(), self.bs, self.path,
                               concurrency=2, verbose=False)
        self.assertTrue(executors[0]._shutdown)
        failed_backend = types.SimpleNamespace(name='fixture', infer=Mock(side_effect=RuntimeError('offline')))
        with patch.object(runner, 'ThreadPoolExecutor', side_effect=create):
            with self.assertRaisesRegex(RuntimeError, 'offline'):
                runner.run_windows(self.windows, failed_backend, self.bs,
                                   self.path.with_name('error.jsonl'), concurrency=2, verbose=False)
        self.assertTrue(executors[1]._shutdown)


if __name__ == '__main__':
    unittest.main()
