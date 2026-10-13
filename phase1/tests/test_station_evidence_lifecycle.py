"""Retained station alarm evidence must distinguish withdrawal and resolution."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import test_workstation_api as fixtures
from webui import server as S


class StationEvidenceLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.WorkstationApiTests()
        self.fixture.setUp()

    def tearDown(self):
        self.fixture.tearDown()

    def test_withdrawn_alarm_is_not_reported_as_a_behavior_hit(self):
        report = self.fixture.start()
        alarm = copy.deepcopy(next(a for a in report['alerts'] if a['behavior_id'] == 'station_away'))
        alarm.update(review_status='rejected', lifecycle='withdrawn',
                     retraction_reason='最新画面证实人员仍在工位', reason='旧几何离岗判断')
        report['alerts'] = [alarm]
        report['alarm_history'] = [alarm]
        S.station_store().save(self.fixture.job_id, report['analysis_revision'], report['scene'], report)
        S.STATE.jobs.clear()
        response = self.fixture.client.get('/api/evidence/' + self.fixture.job_id, params={
            'window_id': 'interval_' + alarm['interval_id'], 'behavior_id': 'station_away'})
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()['model_results']['station_away']
        self.assertFalse(result['violation'], 'Withdrawn evidence cannot still show a behavior hit')
        self.assertEqual(result['review_status'], 'rejected')
        self.assertEqual(result['lifecycle'], 'withdrawn')
        self.assertEqual(result['observable'], alarm['retraction_reason'])

    def test_confirmed_return_keeps_historical_hit_with_recovery_status(self):
        report = self.fixture.start()
        alarm = next(a for a in report['alerts'] if a['behavior_id'] == 'station_away')
        response = self.fixture.client.get('/api/evidence/' + self.fixture.job_id, params={
            'window_id': 'interval_' + alarm['interval_id'], 'behavior_id': 'station_away'})
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()['model_results']['station_away']
        self.assertTrue(result['violation'])
        self.assertEqual(result['review_status'], 'confirmed')
        self.assertEqual(result['lifecycle'], 'resolved')
        self.assertEqual(result['resolution_reason'], 'returned')


if __name__ == '__main__':
    unittest.main()
