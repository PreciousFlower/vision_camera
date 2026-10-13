"""Historical identity edits retain an unobservable final view after restart."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import test_workstation_api as fixtures
from webui import server as S
from webui.workstations import StationLedger


class StationHistoryRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.WorkstationApiTests()
        self.fixture.setUp()
        self.client = self.fixture.client

    def tearDown(self):
        self.fixture.tearDown()

    def archive_unobservable(self, *, with_previous=False, legacy=False):
        scene = copy.deepcopy(self.fixture.scene)
        scene['policy']['confirm_sec'] = 0
        ledger = StationLedger(scene)
        pid = 'P004' if with_previous else 'P001'
        for at in range(4, 7):
            ledger.observe(at, [{'person_id':pid, 'bbox':[.1,.1,.3,.9]}], f'f_{at}')
        for at in (7, 8):
            ledger.observe(at, [], f'f_{at}', observable=False)
        ledger.finish()
        if with_previous:
            # Independent, earlier sightings do not overlap the later person's
            # unknown interval, so a human link must retain that interval.
            earlier = StationLedger(scene)
            for at in range(3):
                earlier.observe(at, [{'person_id':'P001', 'bbox':[.1,.1,.3,.9]}], f'f_{at}')
            earlier.finish()
            ledger.people['P001'] = earlier.people['P001']
            ledger.intervals[:0] = earlier.intervals
        job = S.JobRecord('unobservable-history', 'history.mp4', state='stopped',
                          profile=S.profile_registry().load('workstations'),
                          station_scene=ledger.scene)
        S.STATE.jobs[job.job_id] = job
        library = S.video_library()
        library.begin(job)
        library.save(job)
        report = {**ledger.snapshot(include_intervals=True), 'state':'stopped',
                  'profile':job.profile, 'use_mock':False}
        if legacy:
            report.pop('view_observable', None)
        self.assertIsNone(report['observed_people'])
        self.assertIsNone(report['stations'][0]['observed_count'])
        S.station_store().save(job.job_id, job._analysis_revision, ledger.scene, report)
        S.STATE.jobs.clear()
        return job, report, pid

    def assert_unknown_retained(self, response, previous, source, target):
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertFalse(result['view_observable'])
        self.assertIsNone(result['observed_people'])
        self.assertIsNone(result['stations'][0]['observed_count'])
        before = next(p for p in previous['people'] if p['person_id']==source)
        after = next(p for p in result['people'] if p['person_id']==target)
        self.assertEqual(after['unknown_sec'], before['unknown_sec'])
        self.assertEqual(after['current_duration_sec'], before['current_duration_sec'])
        unknown_before = [i for i in previous['intervals']
                          if i['person_id']==source and i['state']=='unknown']
        indexed = {i['interval_id']:i for i in result['intervals']}
        for span in unknown_before:
            restored = indexed[span['interval_id']]
            self.assertEqual((restored['person_id'],restored['start'],restored['end'],restored['state']),
                             (target,span['start'],span['end'],'unknown'))

    def test_restarted_unobservable_history_can_name_person_without_inventing_zero(self):
        job, report, pid = self.archive_unobservable()
        response = self.client.post(f'/api/stations/{job.job_id}/person', json={
            'analysis_revision':job._analysis_revision, 'person_id':pid,
            'name':'员工A', 'station_id':'a'})
        self.assert_unknown_retained(response, report, pid, pid)

    def test_legacy_unobservable_history_can_merge_without_changing_unknown_duration(self):
        job, report, pid = self.archive_unobservable(with_previous=True, legacy=True)
        response = self.client.post(f'/api/stations/{job.job_id}/person', json={
            'analysis_revision':job._analysis_revision, 'person_id':pid,
            'merge_into':'P001'})
        self.assert_unknown_retained(response, report, pid, 'P001')


if __name__ == '__main__':
    unittest.main()
