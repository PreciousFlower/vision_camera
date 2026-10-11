"""Old code/settings must not silently run new analyses or switch ports."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fastapi.testclient import TestClient
from webui import server as S
from civility.config import load_behaviors
from civility.prompts import sanitize


class RuntimeFreshnessTests(unittest.TestCase):
    def test_current_runtime_is_explicit_and_changed_code_is_detected(self):
        with TestClient(S.create_app()) as client:
            state = client.get('/api/state').json()
            self.assertEqual(state['runtime']['protocol'], 3)
            self.assertFalse(state['runtime']['restart_required'])
            with patch.object(S, 'LOADED_SOURCE_SIGNATURE', 'old'):
                self.assertTrue(client.get('/api/state').json()['runtime']['restart_required'])

    def test_stale_code_blocks_new_jobs_and_gpu_start_but_keeps_stop_available(self):
        with patch.object(S, 'LOADED_SOURCE_SIGNATURE', 'old'), TestClient(S.create_app()) as client:
            for path in ('/api/service/start', '/api/live/start', '/api/detect/unknown'):
                self.assertEqual(client.post(path, json={}).status_code, 409)

    def test_changed_settings_require_restart(self):
        with patch.object(S.STATE, 'settings_signature', 'old'):
            self.assertTrue(S.runtime_status()['settings_changed'])

    def test_busy_port_fails_before_cleanup_and_watchdog(self):
        with patch.object(sys, 'argv', ['server', '--port', '8848']), \
                patch.object(S, 'find_free_port', side_effect=RuntimeError('busy')) as probe, \
                patch.object(S, 'start_watchdog') as watchdog:
            self.assertEqual(S.main(), 2)
            probe.assert_called_once_with(8848, tries=1)
            watchdog.assert_not_called()

    def test_screenshot_raw_model_claim_is_accepted_with_current_definition(self):
        bs = load_behaviors()
        self.assertNotIn('abandoned', bs.by_id('littering').evidence_checks)
        obj = {'scene': '一人在公园小径上丢弃杯子', 'risks': [], 'violations': [
            {'behavior_id': b.id, 'violation': b.id == 'littering',
             **({'confidence': .9, 'observable': '杯从手部脱离后落到地面',
                 'evidence': {'held': 1, 'released': 2, 'landed': 3},
                 'alert': True, 'alert_reason': '明确丢弃垃圾，即使尚未離开也应记录'}
                if b.id == 'littering' else {'alert': False})} for b in bs.enabled]}
        hit = sanitize(obj, bs, frame_count=4)['violations']['littering']
        self.assertTrue(hit['violation'])
        self.assertTrue(hit['alert'])
        self.assertEqual(hit['evidence_issues'], [])

    def test_file_analysis_requests_person_action_sequence_by_default(self):
        job = S.JobRecord('file', 'clip.mp4')
        with patch.object(S.STATE, 'jobs', {'file': job}), \
                patch.object(S, 'run_job') as runner, TestClient(S.create_app()) as client:
            response = client.post('/api/detect/file', json={'use_mock': True})
            self.assertEqual(response.status_code, 200)
            self.assertTrue(job.with_actions)



if __name__ == '__main__':
    unittest.main()
