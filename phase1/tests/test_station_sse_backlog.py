"""Reconnect replays cannot overwrite a workstation's latest alarm history."""
import copy
import json
from pathlib import Path
import queue
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import test_workstation_api as fixtures
from webui import server as S


class StationSseBacklogTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.WorkstationApiTests()
        self.fixture.setUp()

    def tearDown(self):
        self.fixture.tearDown()

    def test_latest_hello_history_survives_old_queued_alert_snapshot(self):
        report = self.fixture.start()
        job = S.STATE.jobs[self.fixture.job_id]
        self.assertGreaterEqual(len(report['alerts']), 2)
        latest = copy.deepcopy(report['alerts'])
        old = copy.deepcopy(latest[:1])
        old[0].update(review_status='confirmed', lifecycle='active')
        latest[0].update(review_status='rejected', lifecycle='withdrawn',
                         reason='最新画面否定旧告警', retraction_reason='最新画面否定旧告警')
        job._station_ledger.alerts = copy.deepcopy(latest)
        job._station_ledger.aliases = {'P004': 'P001'}
        job.alerts = copy.deepcopy(latest)
        job.events = queue.Queue()
        window = {'type': 'window', 'window_id': 'queued-window', 't_start': 1., 't_end': 1.,
                  'frame_indices': [2], 'frame_times': [1.], 'scene': 'P004位于工位',
                  'people': [{'appearance': 'P004', 'action': 'at_station', 'changed': 'P004位于区域内'}],
                  'station_review': {'scene': 'P004位于工位', 'observations': [
                      {'person_id': 'P004', 'frame_id': 'f_2', 'state': 'at_station',
                       'station_id': 'a', 'reason': 'P004位于区域内'}], 'stations': []},
                  'raw': '原始模型输出：P004位于工位'}
        job.events.put(copy.deepcopy(window))
        job.events.put({'type': 'station', 'data': {**copy.deepcopy(report), 'alerts': old}})
        job.events.put({'type': 'alerts', 'alerts': old})
        job.events.put({'type': 'done', 'summary': job.summary()})
        response = self.fixture.client.get('/api/job/' + job.job_id + '/stream')
        self.assertEqual(response.status_code, 200, response.text)
        events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith('data: ')]
        hello = next(event for event in events if event['type'] == 'hello')
        replay = next(event for event in events if event['type'] == 'alerts')
        self.assertEqual(len(hello['summary']['alerts']), len(latest))
        self.assertEqual(len(replay['alerts']), len(latest),
                         'An old queued snapshot removed the newly retained alarm')
        self.assertEqual(replay['alerts'][0]['review_status'], 'rejected',
                         'An old queue item resurrected a withdrawn alarm')
        self.assertEqual(replay['alerts'][0]['lifecycle'], 'withdrawn')
        replay_window = next(event for event in events if event['type'] == 'window')
        self.assertEqual(replay_window['people'][0]['appearance'], 'P001')
        self.assertEqual(replay_window['station_review']['observations'][0]['person_id'], 'P001')
        self.assertNotIn('P004', replay_window['scene'])
        self.assertEqual(replay_window['raw'], window['raw'])
        self.assertEqual(replay_window['frame_indices'], window['frame_indices'])
        self.assertEqual(replay_window['frame_times'], window['frame_times'])
        replay_station = next(event for event in events if event['type'] == 'station')
        self.assertEqual(replay_station['data']['alerts'][0]['review_status'], 'rejected')
        self.assertEqual(replay_station['data']['aliases'], {'P004': 'P001'})


if __name__ == '__main__':
    unittest.main()
