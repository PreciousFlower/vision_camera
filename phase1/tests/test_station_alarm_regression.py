"""Alarm clock runs independently of model completion and records actionable times."""
import copy
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from webui.workstations import StationLedger

SCENE={'stations':[{'id':'a','name':'工位A','expected_people':1,'polygon':[[0,0],[.6,0],[.6,1],[0,1]]}],
       'policy':{'confirm_sec':1,'away_sec':5,'empty_sec':5,'review_interval_sec':10}}
INSIDE={'person_id':'P001','bbox':[.1,.1,.4,.8]}
OUTSIDE={'person_id':'P001','bbox':[.72,.1,.95,.8]}


class StationAlarmRegressionTests(unittest.TestCase):
    def test_runner_checks_threshold_between_model_review_completions(self):
        import numpy as np
        from webui import server as S
        from webui.workstation_runner import run_station_job
        job=S.JobRecord('alarm-clock','sample.mp4',profile={'mode':'workstation','behaviors':[{'id':'station_away','enabled':True}]},station_scene=copy.deepcopy(SCENE))
        job._analysis_revision='v1';capture=[]
        image=np.random.default_rng(4).integers(40,180,(80,120,3),dtype=np.uint8)
        def frames(*args):
            for at in range(10):
                if at==8:capture.append(copy.deepcopy(job.alerts))
                frame=image.copy();frame[0,0,0]=at
                yield float(at),frame,10.,None
                time.sleep(.015)
        def detect(frame):return [{'bbox':(INSIDE if int(frame[0,0,0])<2 else OUTSIDE)['bbox']}]
        def review(images,evidence,stations,people,profile):
            latest=evidence[-1];outside=latest['time_sec']>=2
            return {'parsed':{'scene':'真实接口适配器','observations':[{'person_id':p['person_id'],'station_id':'a','state':'outside' if outside else 'at_station','frame_id':latest['frame_id'],'reason':'画面位置核对'} for p in people],
                             'stations':[{'station_id':'a','frame_id':latest['frame_id'],'visible':True,'count':0 if outside else 1,'reason':'人数核对'}]}}
        with tempfile.TemporaryDirectory() as root,patch('webui.workstation_runner.file_frames',side_effect=frames),patch('webui.workstation_runner.detector_for',return_value=SimpleNamespace(detect=detect)):
            run_station_job(job,'ignored',{'paths':{'data_dir':root,'frames_dir':root}},SimpleNamespace(review_workstations=review),Mock(),Mock(),lambda text:None)
        self.assertEqual(job.state,'done',job.error)
        self.assertTrue(capture and any(a['behavior_id']=='station_away' for a in capture[0]),'到7秒阈值时告警必须已经发布，不能等下一次模型复核完成')
    def test_alarm_time_is_threshold_not_late_model_return_time(self):
        ledger=StationLedger(SCENE)
        for t in range(11):ledger.observe(t,copy.deepcopy([INSIDE] if t<2 else [OUTSIDE]),f'f_{t}')
        ledger.review({'observations':[{'person_id':'P001','station_id':'a','state':'outside','frame_id':'f_3','reason':'明确位于工位外'}]},[{'frame_id':'f_3','time_sec':3}])
        alarm=ledger.alarms()[0]
        self.assertEqual(alarm.get('threshold_time_sec'),7)
        self.assertEqual(alarm.get('trigger_time_sec'),10)
        self.assertEqual(alarm.get('verification_delay_sec'),3)
    def test_visible_exit_then_missing_is_departure_not_minutes_unknown(self):
        ledger=StationLedger(SCENE)
        for t in (0,1):ledger.observe(t,[copy.deepcopy(INSIDE)],f'f_{t}')
        ledger.observe(2,[{**OUTSIDE,'departure_evidence':True}], 'f_2')
        for t in range(3,15):ledger.observe(t,[],f'f_{t}')
        person=ledger.snapshot()['people'][0]
        self.assertEqual(person['away_sec'],12)
        self.assertEqual(person['away_count'],1)
        self.assertEqual(person['unknown_sec'],0)
    def test_blind_disappearance_retains_explained_unknown(self):
        ledger=StationLedger(SCENE)
        for t in (0,1):ledger.observe(t,[copy.deepcopy(INSIDE)],f'f_{t}')
        for t in range(2,5):ledger.observe(t,[],f'f_{t}')
        person=ledger.snapshot()['people'][0]
        self.assertEqual(person['away_sec'],0)
        self.assertIn('遮挡',person.get('unknown_reason',''))
    def test_pending_alarm_is_immediate_then_confirmed_without_duplicate(self):
        ledger=StationLedger(SCENE,enabled=['station_away'])
        for t in range(8):ledger.observe(t,copy.deepcopy([INSIDE] if t<2 else [OUTSIDE]),f'f_{t}')
        alarm=ledger.alarms()[0]
        self.assertEqual((alarm['threshold_time_sec'],alarm['trigger_time_sec'],alarm['review_status']),(7,7,'pending'))
        self.assertEqual(ledger.alarms(),[])
        ledger.observe(8,[copy.deepcopy(OUTSIDE)],'f_8')
        ledger.review({'observations':[{'person_id':'P001','station_id':'a','state':'outside','frame_id':'f_8','reason':'真实画面确认位于区域外'}]},[{'frame_id':'f_8','time_sec':8}])
        update=ledger.alarms()[0]
        self.assertEqual(update['review_status'],'confirmed');self.assertEqual(update['trigger_time_sec'],7)
        self.assertEqual(update['verification_delay_sec'],1);self.assertEqual(len(ledger.alerts),1)
    def test_model_rejects_pending_alarm_and_keeps_audit_history(self):
        ledger=StationLedger(SCENE,enabled=['station_away'])
        for t in range(8):ledger.observe(t,copy.deepcopy([INSIDE] if t<2 else [OUTSIDE]),f'f_{t}')
        self.assertEqual(ledger.alarms()[0]['review_status'],'pending')
        ledger.review({'observations':[{'person_id':'P001','station_id':'a','state':'at_station','frame_id':'f_7','reason':'仍在原工位，位置检测不一致'}]},[{'frame_id':'f_7','time_sec':7}])
        self.assertEqual(ledger.alarms()[0]['review_status'],'rejected')
        self.assertEqual(len(ledger.snapshot()['alerts']),1)
        self.assertEqual(ledger.snapshot()['alerts'][0]['lifecycle'],'withdrawn')
        self.assertEqual(ledger.snapshot()['alarm_history'][0]['review_status'],'rejected')
    def test_single_station_empty_after_grace_is_away_and_model_can_confirm(self):
        ledger=StationLedger(SCENE)
        for t in (0,1):ledger.observe(t,[copy.deepcopy(INSIDE)],f'f_{t}')
        for t in range(2,8):ledger.observe(t,[],f'f_{t}')
        person=ledger.snapshot()['people'][0]
        self.assertEqual((person['away_sec'],person['unknown_sec'],person['away_count']),(5,0,1))
        updates=ledger.alarms();self.assertTrue(any(a['behavior_id']=='station_away' and a['review_status']=='pending' for a in updates))
        ledger.review({'stations':[{'station_id':'a','visible':True,'count':0,'frame_id':'f_7','reason':'工位区域可观察，确实为空'}]},[{'frame_id':'f_7','time_sec':7}])
        self.assertTrue(any(a['behavior_id']=='station_away' and a['review_status']=='confirmed' for a in ledger.alarms()))
    def test_model_occluded_station_cannot_create_departure_alarm(self):
        ledger=StationLedger(SCENE)
        for t in (0,1):ledger.observe(t,[copy.deepcopy(INSIDE)],f'f_{t}')
        ledger.observe(2,[],'f_2')
        ledger.review({'stations':[{'station_id':'a','visible':False,'count':None,'frame_id':'f_2','reason':'工位被遮挡'}]},[{'frame_id':'f_2','time_sec':2}])
        for t in range(3,10):ledger.observe(t,[],f'f_{t}')
        self.assertEqual(ledger.snapshot()['people'][0]['away_sec'],0)
        self.assertEqual(ledger.alarms(),[])
    def test_complete_single_person_departure_return_uses_one_id_and_clock(self):
        import numpy as np
        from webui import server as S
        from webui.workstation_runner import run_station_job
        scene=copy.deepcopy(SCENE);scene['policy'].update(confirm_sec=3,away_sec=10,empty_sec=60)
        job=S.JobRecord('return-clock','sample.mp4',use_mock=True,profile={'mode':'workstation','behaviors':[{'id':'station_away','enabled':True}]},station_scene=scene)
        job._analysis_revision='v1';captured=[]
        image=np.random.default_rng(4).integers(40,180,(80,120,3),dtype=np.uint8)
        def frames(*args):
            for step in range(161):
                at=step/2
                if at==15.5:captured.append(copy.deepcopy(job.alerts))
                frame=image.copy();frame[0,0,0]=int(at)
                yield at,frame,80.,None
                time.sleep(.001)
        def detect(frame):
            at=int(frame[0,0,0]);return [] if 5<=at<70 else [{'bbox':INSIDE['bbox'],'_appearance':[1.,0.]}]
        with tempfile.TemporaryDirectory() as root,patch('webui.workstation_runner.file_frames',side_effect=frames),patch('webui.workstation_runner.detector_for',return_value=SimpleNamespace(detect=detect)):
            run_station_job(job,'ignored',{'paths':{'data_dir':root,'frames_dir':root}},None,Mock(),Mock(),lambda text:None)
        self.assertEqual(job.state,'done',job.error);report=job._station_ledger.snapshot()
        self.assertEqual(len(report['people']),1);person=report['people'][0]
        self.assertEqual((person['person_id'],person['away_count'],person['away_sec'],person['unknown_sec']),('P001',1,65,0))
        self.assertTrue(captured[0]);self.assertEqual(captured[0][0]['trigger_time_sec'],15)
    def test_one_second_threshold_is_not_delayed_by_three_second_grace(self):
        scene=copy.deepcopy(SCENE);scene['policy'].update(confirm_sec=3,occlusion_sec=3,away_sec=1)
        for missing in (False,True):
            ledger=StationLedger(scene,enabled=['station_away'])
            for t in range(4):ledger.observe(t,[copy.deepcopy(INSIDE)],f'f_{t}')
            ledger.observe(4,[] if missing else [copy.deepcopy(OUTSIDE)],'f_4')
            ledger.observe(5,[] if missing else [copy.deepcopy(OUTSIDE)],'f_5')
            self.assertEqual(ledger.alarms()[0]['trigger_time_sec'],5)


if __name__=='__main__':unittest.main()
