"""Actual HTTP/manual-merge and delayed-runner regressions in isolated storage."""
import copy
import json
from pathlib import Path
import sys
import threading
import unittest
from unittest.mock import patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import test_workstation_api as fixtures
from webui import server as S
from webui.workstations import StationLedger


class RawIdDetector:
    def detect(self, image):
        marker = int(image[0, 0, 0])
        if marker not in (1, 4):
            return []
        return [{'person_id':f'P{marker:03d}', 'bbox':[.1,.1,.3,.9], 'confidence':.99}]


class RawIdTracker:
    def __init__(self, **kwargs):
        pass
    def link_identities(self, aliases):
        pass
    def update(self, timestamp, detections):
        return copy.deepcopy(detections)


class StationIdentityApiTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.WorkstationApiTests()
        self.fixture.setUp()
        self.client = self.fixture.client
        self.released = threading.Event()
    def tearDown(self):
        self.released.set()
        self.fixture.tearDown()
    def archive(self):
        job = S.JobRecord('identity-history', 'history.mp4', state='done',
                          profile=S.profile_registry().load('workstations'),
                          station_scene=self.fixture.scene, windows_seen=1,
                          windows_done=1)
        ledger = StationLedger(self.fixture.scene)
        for t in range(10):
            pid = 'P001' if t < 2 else 'P003' if 4 <= t < 6 else 'P004' if t >= 8 else None
            ledger.observe(t, [{'person_id':pid,'bbox':[.1,.1,.3,.9]}] if pid else [], f'f_{t}')
        job._station_ledger = ledger
        S.STATE.jobs[job.job_id] = job
        library = S.video_library()
        library.begin(job)
        row = {'window_id':job.job_id+'__w19', 't_start':8, 't_end':9, 'frames':1,
               'frame_indices':[9], 'frame_times':[9], 'frame_labels':['t=9.0s'],
               'scene':'1号工位有人员P004驻留', 'people':[
                   {'where':'a','appearance':'P004','action':'at_station','changed':'P004身体中心在区域内'}],
               'station_review':{'scene':'1号工位有人员P004驻留','observations':[
                   {'person_id':'P004','station_id':'a','frame_id':'f_9','state':'at_station','reason':'P004身体中心在区域内'}],
                   'stations':[{'station_id':'a','frame_id':'f_9','visible':True,'count':1,'reason':'P004在内，P003不在区域内'}]},
               'raw':'原始模型：P004在岗，P003离开', 'error':None, 'violations':{},'risks':[]}
        job.window_results = [row]
        library.record(job, row)
        library.save(job)
        S.station_store().save(job.job_id, job._analysis_revision, ledger.scene,
            {**ledger.snapshot(include_intervals=True),'state':'done','profile':job.profile,'use_mock':False})
        return job
    def merge(self, job, source):
        response = self.client.post(f'/api/stations/{job.job_id}/person', json={
            'analysis_revision':job._analysis_revision,'person_id':source,'merge_into':'P001'})
        self.assertEqual(response.status_code, 200, response.text)
    def assert_windows_canonical(self, response):
        self.assertEqual(response.status_code, 200, response.text)
        rows = response.json()['window_results']
        row = next(row for row in rows if '原始模型：P004' in row.get('raw',''))
        self.assertEqual(row['people'][0]['appearance'], 'P001')
        self.assertIn('P001', row['scene'])
        self.assertNotIn('P004', row['scene'])
        self.assertNotIn('P003', row['station_review']['stations'][0]['reason'])
        self.assertEqual(row['station_review']['observations'][0]['person_id'], 'P001')
        self.assertIn('P004', row['raw'])
        return row
    def test_manual_merge_updates_job_and_archive_display_without_rewriting_raw(self):
        job = self.archive()
        self.merge(job, 'P003')
        self.merge(job, 'P004')
        self.assert_windows_canonical(self.client.get(f'/api/job/{job.job_id}'))
        self.assert_windows_canonical(self.client.get(f'/api/videos/{job.job_id}/analysis'))
        archived = S.video_library().windows(job.job_id)[0]
        self.assertEqual(archived['people'][0]['appearance'], 'P004')
        self.assertIn('P004', archived['raw'])
    def test_restarted_history_uses_persisted_aliases(self):
        job = self.archive()
        self.merge(job, 'P003')
        self.merge(job, 'P004')
        S.STATE.jobs.clear()
        self.assert_windows_canonical(self.client.get(f'/api/job/{job.job_id}'))
        self.assert_windows_canonical(self.client.get(f'/api/videos/{job.job_id}/analysis'))
    def test_existing_alarm_reason_follows_manual_link_and_keeps_original_audit(self):
        job = self.archive()
        alert = {'alarm_id':'saved-departure','interval_id':'saved-departure','person_id':'P003',
                 'station_id':'a','behavior_id':'station_away','behavior_name':'离岗超时','source':'workstation',
                 'severity':'medium','t_start':4.,'t_end':6.,'duration_sec':2.,'review_status':'confirmed',
                 'reason':'P003已明确离开工位', 'latest_review_reason':'P003不在蓝色区域内',
                 'status_history':[{'review_status':'confirmed','reason':'P003已明确离开工位'}]}
        job.alerts = [copy.deepcopy(alert)]
        job._station_ledger.alerts = [copy.deepcopy(alert)]
        S.video_library().save(job)
        S.station_store().save(job.job_id, job._analysis_revision, job.station_scene,
            {**job._station_ledger.snapshot(include_intervals=True),'state':'done','profile':job.profile,'use_mock':False})
        self.merge(job, 'P003')
        self.merge(job, 'P004')
        for route in (f'/api/stations/{job.job_id}', f'/api/job/{job.job_id}', f'/api/videos/{job.job_id}/analysis'):
            response = self.client.get(route)
            self.assertEqual(response.status_code, 200, response.text)
            updated = next(a for a in response.json()['alerts'] if a['alarm_id']=='saved-departure')
            self.assertEqual(updated['person_id'], 'P001', route)
            self.assertIn('P001', updated['reason'], route)
            self.assertNotIn('P003', updated['reason'], route)
            self.assertIn('P003', updated['original_reason'], route)
            self.assertIn('P003', updated['status_history'][0]['reason'], route)
    def test_delayed_model_reply_after_merge_cannot_revive_old_id(self):
        self.fixture.upload()
        requested = threading.Event()
        submitted = []
        def frames(source, fps, stopped):
            for t in range(11):
                if t == 9:
                    if not requested.wait(5):
                        raise AssertionError('未进入含P004的异步模型请求')
                    if not self.released.wait(5):
                        raise AssertionError('人工关联未完成')
                image = np.full((270,480,3), 225, np.uint8)
                cv2.rectangle(image,(45,25),(140,245),(80,110,160),-1)
                image[0,0] = 1 if t < 2 else 4 if t >= 4 else 0
                yield float(t), image, 11., None
        def review(images, evidence, stations, people, profile):
            submitted.append(copy.deepcopy({'evidence':evidence,'people':people}))
            pid = 'P004' if any(p['person_id']=='P004' for p in people) else 'P001'
            if pid == 'P004':
                requested.set()
                if not self.released.wait(5):
                    raise AssertionError('异步模型复核未释放')
            frame = evidence[-1]['frame_id']
            return {'parsed':{'scene':pid+'位于工位区域', 'observations':[
                {'person_id':pid,'station_id':'a','frame_id':frame,'state':'at_station','reason':pid+'身体中心位于蓝色区域'}],
                'stations':[{'station_id':'a','frame_id':frame,'visible':True,'count':1,'reason':pid+'在区域内'}]},
                'raw':'原始模型：'+pid+'在岗', 'model':'isolated-model'}
        self.fixture.session.review_workstations = review
        with patch('webui.workstation_runner.file_frames',side_effect=frames), \
             patch('webui.workstation_runner.detector_for',return_value=RawIdDetector()), \
             patch('webui.workstation_runner.AnonymousTracker',RawIdTracker):
            response = self.client.post('/api/detect/'+self.fixture.job_id, json={
                'profile_id':'workstations','station_scene':self.fixture.scene,'use_mock':False})
            self.assertEqual(response.status_code, 200, response.text)
            job = S.STATE.jobs[self.fixture.job_id]
            try:
                self.assertTrue(requested.wait(5), '未进入异步P004模型复核')
                self.merge(job, 'P004')
            finally:
                self.released.set()
            job._detect_worker.join(10)
            self.assertFalse(job._detect_worker.is_alive())
            self.assertEqual(job.state, 'done', job.error)
        response = self.client.get(f'/api/job/{job.job_id}')
        self.assert_windows_canonical(response)
        self.assertEqual([p['person_id'] for p in job._station_ledger.snapshot()['people']], ['P001'])
        # A later request must send canonical geometry as well as canonical
        # people; otherwise the model sees contradictory snapshot labels.
        last = submitted[-1]
        self.assertEqual([p['person_id'] for p in last['people']], ['P001'])
        hints = [h['person_id'] for frame in last['evidence'] for h in frame.get('geometry_hint',[])]
        self.assertTrue(hints)
        self.assertEqual(set(hints), {'P001'})


if __name__ == '__main__':
    unittest.main()
