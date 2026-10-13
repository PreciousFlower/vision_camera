"""A concurrent manual link must not split a model request's identity snapshot."""
import copy
import inspect
from pathlib import Path
import sys
import threading
import unittest
from unittest.mock import patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import test_station_identity_api as fixtures
from webui import server as S
from webui.workstations import StationLedger


class StationSubmissionAtomicTests(unittest.TestCase):
    def test_other_thread_manual_merge_keeps_latest_geometry_and_roster_coherent(self):
        fixture = fixtures.StationIdentityApiTests()
        fixture.setUp()
        merge_requested = threading.Event()
        merge_completed = threading.Event()
        submitted = []
        merge_errors = []
        source_ledger = []
        worker = None
        original_snapshot = StationLedger.snapshot

        def merge_person():
            if not merge_requested.wait(5):
                return
            try:
                source_ledger[0].merge('P004', 'P001')
            except Exception as exc:
                merge_errors.append(exc)
            finally:
                merge_completed.set()

        def snapshot(ledger, *args, **kwargs):
            caller = inspect.currentframe().f_back.f_code.co_name
            if caller == 'submit' and 'P004' in ledger.people and not merge_requested.is_set():
                source_ledger.append(ledger)
                merge_requested.set()
                # With an atomic submission lock the other thread waits, and
                # this call returns its consistent pre-merge snapshot. Without
                # that lock it finishes the merge before the roster is read.
                merge_completed.wait(.3)
            return original_snapshot(ledger, *args, **kwargs)

        def frames(source, fps, stopped):
            for t in range(10):
                image = np.full((270, 480, 3), 225, np.uint8)
                cv2.rectangle(image, (45, 25), (140, 245), (80, 110, 160), -1)
                image[0, 0] = 1 if t < 2 else 4 if t >= 4 else 0
                yield float(t), image, 10., None

        def review(images, evidence, stations, people, profile):
            submitted.append(copy.deepcopy({'evidence': evidence, 'people': people}))
            frame_id = evidence[-1]['frame_id']
            return {'parsed': {'scene': '区域内有人', 'observations': [], 'stations': [
                {'station_id': 'a', 'frame_id': frame_id, 'visible': True,
                 'count': 1, 'reason': '区域内有人'}]}, 'raw': '原始人数核对'}

        try:
            fixture.fixture.upload()
            fixture.fixture.session.review_workstations = review
            worker = threading.Thread(target=merge_person, name='isolated-person-link')
            worker.start()
            with patch('webui.workstation_runner.file_frames', side_effect=frames), \
                 patch('webui.workstation_runner.detector_for', return_value=fixtures.RawIdDetector()), \
                 patch('webui.workstation_runner.AnonymousTracker', fixtures.RawIdTracker), \
                 patch.object(StationLedger, 'snapshot', snapshot):
                response = fixture.client.post('/api/detect/' + fixture.fixture.job_id, json={
                    'profile_id': 'workstations', 'station_scene': fixture.fixture.scene, 'use_mock': False})
                self.assertEqual(response.status_code, 200, response.text)
                job = S.STATE.jobs[fixture.fixture.job_id]
                job._detect_worker.join(10)
                self.assertFalse(job._detect_worker.is_alive())
                self.assertEqual(job.state, 'done', job.error)
            worker.join(5)
            self.assertTrue(merge_requested.is_set(), 'Concurrent merge boundary was not reached')
            self.assertTrue(merge_completed.is_set(), 'Manual identity link did not complete')
            self.assertEqual(merge_errors, [])
            relevant = [request for request in submitted
                        if request['evidence'][-1]['time_sec'] >= 4]
            self.assertTrue(relevant)
            for request in relevant:
                roster = {person['person_id'] for person in request['people']}
                hints = request['evidence'][-1]['geometry_hint']
                self.assertEqual(len(hints), 1,
                    'Latest visible person was dropped by a mixed pre/post-merge request')
                self.assertTrue({hint['person_id'] for hint in hints} <= roster)
        finally:
            if not merge_requested.is_set():
                merge_requested.set()
            if worker:
                worker.join(5)
            fixture.tearDown()


if __name__ == '__main__':
    unittest.main()
