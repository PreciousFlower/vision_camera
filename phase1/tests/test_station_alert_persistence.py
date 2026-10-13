"""Alert history remains visible after return, uncertainty or explicit rejection."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from webui.workstations import StationLedger, StationStore


SCENE = {'stations': [{'id': 'a', 'name': '工位A', 'expected_people': 1,
                      'polygon': [[0, 0], [.6, 0], [.6, 1], [0, 1]]}],
         'policy': {'confirm_sec': 0, 'occlusion_sec': 1, 'away_sec': 2,
                    'empty_sec': 2, 'review_interval_sec': 5}}
INSIDE = {'person_id': 'P001', 'bbox': [.1, .1, .4, .8]}
OUTSIDE = {'person_id': 'P001', 'bbox': [.72, .1, .95, .8]}


class StationAlertPersistenceTests(unittest.TestCase):
    def make_departure(self, *, missing=False):
        ledger = StationLedger(SCENE, enabled=['station_away'])
        for at in range(5):
            tracks = [INSIDE] if at < 2 else [] if missing else [OUTSIDE]
            ledger.observe(at, copy.deepcopy(tracks), f'f_{at}')
        ledger.alarms()
        self.check(ledger, 4, count=0, state=None if missing else 'outside')
        ledger.alarms()
        self.assertEqual(ledger.snapshot()['alerts'][0]['review_status'], 'confirmed')
        return ledger

    def check(self, ledger, at, count=None, state=None):
        result = {'stations': [], 'observations': []}
        if count is not None:
            result['stations'].append({'station_id': 'a', 'visible': True,
                                       'count': count, 'frame_id': f'f_{at}',
                                       'reason': '画面核对：空工位' if count == 0 else '画面核对：工位有人'})
        if state is not None:
            result['observations'].append({'person_id': 'P001', 'station_id': 'a',
                                           'state': state, 'frame_id': f'f_{at}',
                                           'reason': '明确在区域外' if state == 'outside' else '位置暂无法确认'})
        ledger.review(result, [{'frame_id': f'f_{at}', 'time_sec': at}])

    def test_rejected_alarm_remains_visible_with_reason_and_time(self):
        ledger = self.make_departure(missing=True)
        # A later result explicitly contradicts the inferred departure.
        self.check(ledger, 4, count=1)
        updates = ledger.alarms()
        self.assertEqual(updates[0]['review_status'], 'rejected')
        alerts = ledger.snapshot()['alerts']
        self.assertEqual(len(alerts), 1, '撤回必须保留在现场告警，不能静默消失')
        self.assertEqual(alerts[0]['lifecycle'], 'withdrawn')
        self.assertEqual(alerts[0]['retracted_time_sec'], 4)
        self.assertIn('工位有人', alerts[0]['retraction_reason'])
        self.assertEqual([x['review_status'] for x in alerts[0]['status_history']],
                         ['pending', 'confirmed', 'rejected'])

    def test_return_resolves_alert_and_freezes_actual_closed_duration(self):
        ledger = self.make_departure(missing=True)
        ledger.observe(5, [copy.deepcopy(INSIDE)], 'f_5')
        self.check(ledger, 5, count=1)
        updates = ledger.alarms()
        self.assertTrue(updates, '恢复状态必须推送给现场告警')
        alert = ledger.snapshot()['alerts'][0]
        self.assertEqual((alert['review_status'], alert['lifecycle']), ('confirmed', 'resolved'))
        self.assertEqual((alert['t_end'], alert['duration_sec'], alert['resolved_time_sec']), (5, 3, 5))
        ledger.observe(6, [copy.deepcopy(INSIDE)], 'f_6')
        self.assertEqual(ledger.alarms(), [])
        self.assertEqual(ledger.snapshot()['alerts'][0]['duration_sec'], 3)

    def test_later_uncertain_observation_does_not_erase_confirmed_past_departure(self):
        ledger = self.make_departure()
        self.check(ledger, 4, state='outside')
        ledger.alarms()
        ledger.observe(5, [copy.deepcopy(OUTSIDE)], 'f_5')
        self.check(ledger, 5, state='uncertain')
        updates = ledger.alarms()
        self.assertEqual(updates[0]['latest_review_status'], 'uncertain')
        alert = ledger.snapshot()['alerts'][0]
        self.assertEqual(alert['review_status'], 'confirmed')
        self.assertEqual(alert['latest_review_status'], 'uncertain')
        self.assertIn('位置暂无法确认', alert['latest_review_reason'])
        self.assertNotIn('位置暂无法确认', alert['reason'])

    def test_review_expiry_does_not_downgrade_confirmed_alert(self):
        ledger = self.make_departure()
        self.check(ledger, 4, state='outside')
        ledger.alarms()
        for at in range(5, 36):
            ledger.observe(at, [copy.deepcopy(OUTSIDE)], f'f_{at}')
            ledger.alarms()
        alert = ledger.snapshot()['alerts'][0]
        self.assertEqual(alert['review_status'], 'confirmed')
        self.assertEqual(alert['latest_review_status'], 'stale')
        self.assertEqual(alert['duration_sec'], 33)
        self.assertEqual(len(ledger.alerts), 1)

    def test_legacy_filtered_history_load_restores_withdrawn_alarm_without_fake_time(self):
        ledger = self.make_departure(missing=True)
        self.check(ledger, 4, count=1)
        ledger.alarms()
        summary = ledger.snapshot(include_intervals=True)
        summary['alerts'] = []
        for field in ('lifecycle', 'retracted_time_sec', 'retraction_reason', 'status_history'):
            summary['alarm_history'][0].pop(field, None)
        with tempfile.TemporaryDirectory() as directory:
            store = StationStore(Path(directory))
            store.save('job', 'revision', ledger.scene, summary)
            result = store.load('job', 'revision')
        self.assertEqual(len(result['alerts']), 1)
        self.assertEqual(result['alerts'][0]['lifecycle'], 'withdrawn')
        self.assertIsNone(result['alerts'][0]['retracted_time_sec'])
        self.assertIn('工位有人', result['alerts'][0]['retraction_reason'])

    def test_legacy_missing_review_status_never_invents_model_confirmation(self):
        from webui.workstations import restore_alarm_history
        alert = {'alarm_id': 'legacy', 'interval_id': 'span', 'behavior_id': 'station_away',
                 't_start': 2, 't_end': 8, 'trigger_time_sec': 5, 'reason': '旧告警'}
        summary = {'alerts': [alert], 'intervals': [{'interval_id': 'span', 'end': 8}]}
        restored = restore_alarm_history(summary)
        self.assertEqual(restored['alerts'][0]['review_status'], 'unknown')
        for status in ('confirmed', 'rejected'):
            summary['intervals'][0]['review_status'] = status
            self.assertEqual(restore_alarm_history(summary)['alerts'][0]['review_status'], status)

    def test_uncertain_station_review_updates_existing_alert_but_keeps_confirmation(self):
        ledger = StationLedger(SCENE, enabled=['station_empty'])
        for at in range(3):
            ledger.observe(at, [], f'f_{at}')
        ledger.alarms()
        self.check(ledger, 2, count=0)
        ledger.alarms()
        ledger.observe(3, [], 'f_3')
        ledger.review({'stations': [{'station_id': 'a', 'visible': False, 'count': None,
                                     'frame_id': 'f_3', 'reason': '当前区域被遮挡'}]},
                      [{'frame_id': 'f_3', 'time_sec': 3}])
        updates = ledger.alarms()
        self.assertEqual(updates[0]['latest_review_status'], 'uncertain')
        self.assertEqual(updates[0]['review_status'], 'confirmed')
        self.assertIn('当前区域被遮挡', updates[0]['latest_review_reason'])

    def test_capture_gap_closes_event_without_claiming_the_person_returned(self):
        ledger = self.make_departure()
        ledger.observe(10, [copy.deepcopy(OUTSIDE)], 'f_10')
        ledger.alarms()
        alert = ledger.snapshot()['alerts'][0]
        self.assertEqual(alert['lifecycle'], 'resolved')
        self.assertEqual(alert['resolution_reason'], 'observation_lost')

    def test_finished_away_event_records_monitoring_end_not_return(self):
        ledger = self.make_departure()
        ledger.finish()
        ledger.alarms()
        alert = ledger.snapshot()['alerts'][0]
        self.assertEqual(alert['lifecycle'], 'resolved')
        self.assertEqual(alert['resolution_reason'], 'session_ended')


class StationAlertArchiveApiTests(unittest.TestCase):
    def setUp(self):
        from test_workstation_api import WorkstationApiTests
        self.fixture = WorkstationApiTests()
        self.fixture.setUp()

    def tearDown(self):
        self.fixture.tearDown()

    def test_history_analysis_and_manual_name_preserve_old_withdrawn_alarm(self):
        from webui import server as S
        fixture = self.fixture
        report = fixture.start()
        alarm = next(a for a in report['alerts'] if a['behavior_id'] == 'station_away')
        # Simulate a summary written by the previous filtered-alerts version.
        alarm = {**alarm, 'review_status': 'rejected', 'reason': '模型复核否定原离岗判断'}
        for field in ('lifecycle', 'retracted_time_sec', 'retraction_reason', 'status_history'):
            alarm.pop(field, None)
        report['alerts'] = []
        report['alarm_history'] = [alarm]
        S.station_store().save(fixture.job_id, report['analysis_revision'], report['scene'], report)
        S.STATE.jobs.clear()
        restored = fixture.client.get('/api/stations/' + fixture.job_id).json()
        self.assertEqual(restored['alerts'][0]['lifecycle'], 'withdrawn')
        history = fixture.client.get(f'/api/videos/{fixture.job_id}/analysis')
        self.assertEqual(history.status_code, 200, history.text)
        self.assertEqual(history.json()['alerts'][0]['review_status'], 'rejected')
        renamed = fixture.client.post(f'/api/stations/{fixture.job_id}/person',
                                     json={'analysis_revision': report['analysis_revision'],
                                           'person_id': 'P001', 'name': '员工A', 'station_id': 'a'})
        self.assertEqual(renamed.status_code, 200, renamed.text)
        self.assertEqual(renamed.json()['alarm_history'][0]['lifecycle'], 'withdrawn')


if __name__ == '__main__':
    unittest.main()
