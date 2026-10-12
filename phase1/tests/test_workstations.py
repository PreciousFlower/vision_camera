"""Chronological station metrics and visual review contracts, without a GPU."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from webui.workstations import StationLedger, StationStore, validate_scene
from webui.person_tracking import AnonymousTracker
from webui.behavior_profiles import ProfileRegistry
from civility.station_review import validate_request,validate_result


SCENE={'stations':[{'id':'a','name':'1号工位','polygon':[[0,0],[.45,0],[.45,1],[0,1]],'expected_people':1}],
       'policy':{'confirm_sec':1,'away_sec':2,'empty_sec':2,'review_interval_sec':5}}
INSIDE={'person_id':'P001','bbox':[.1,.1,.3,.9]}
OUTSIDE={'person_id':'P001','bbox':[.7,.1,.9,.9]}


class LedgerTests(unittest.TestCase):
    def setUp(self): self.ledger=StationLedger(SCENE)
    def tick(self,t,tracks): self.ledger.observe(t,copy.deepcopy(tracks),f'f_{int(t)}')
    def person(self): return self.ledger.snapshot()['people'][0]
    def test_enter_leave_return_durations_are_capture_time_and_no_overlap(self):
        for t,tracks in [(0,[INSIDE]),(1,[INSIDE]),(2,[OUTSIDE]),(3,[OUTSIDE]),(4,[INSIDE]),(5,[INSIDE])]: self.tick(t,tracks)
        person=self.person()
        self.assertEqual((person['at_station_sec'],person['away_sec'],person['unknown_sec']),(3,2,0))
        self.assertEqual(person['away_count'],1);self.assertEqual(person['longest_away_sec'],2)
        self.assertEqual(person['current_duration_sec'],1)
        spans=self.ledger.snapshot(include_intervals=True)['intervals']
        for a,b in zip(spans,spans[1:]): self.assertLessEqual(a['end'],b['start'])
    def test_occlusion_is_unknown_and_does_not_trigger_departure(self):
        for t,tracks in [(0,[INSIDE]),(1,[INSIDE]),(2,[]),(3,[]),(4,[INSIDE]),(5,[INSIDE])]: self.tick(t,tracks)
        self.assertEqual(self.person()['away_sec'],0);self.assertEqual(self.person()['unknown_sec'],2)
        self.assertEqual(self.ledger.alarms(),[])
    def test_capture_gap_is_unknown_even_if_person_seen_outside_afterward(self):
        self.tick(0,[INSIDE]);self.tick(1,[INSIDE]);self.tick(10,[OUTSIDE]);self.tick(11,[OUTSIDE])
        person=self.person();self.assertEqual(person['unknown_sec'],9);self.assertEqual(person['away_sec'],1)
        self.assertEqual(self.ledger.snapshot()['gaps'],[{'start':1,'end':10}])
    def test_people_simultaneous_and_overlapping_regions_do_not_double_count(self):
        scene=copy.deepcopy(SCENE);scene['stations'].append({**scene['stations'][0],'id':'b','name':'重叠工位'})
        ledger=StationLedger(scene);tracks=[INSIDE,{'person_id':'P002','bbox':[.31,.1,.43,.9]}]
        ledger.observe(0,copy.deepcopy(tracks),'f_0');ledger.observe(1,copy.deepcopy(tracks),'f_1')
        result=ledger.snapshot();self.assertEqual(result['observed_people'],2);self.assertEqual(result['resident_people'],2)
        self.assertEqual(sum(s['observed_count'] for s in result['stations']),2)
    def test_departure_alarm_requires_current_model_evidence_and_is_deduplicated(self):
        for t in (0,1): self.tick(t,[INSIDE])
        for t in (2,3,4): self.tick(t,[OUTSIDE])
        self.assertEqual(self.ledger.alarms(),[])
        self.ledger.review({'observations':[{'person_id':'P001','state':'outside','station_id':'a','frame_id':'f_4','reason':'确实离开工位'}],'stations':[]},[{'frame_id':'f_4','time_sec':4}])
        result=self.ledger.alarms();self.assertEqual(result[0]['duration_sec'],2);self.assertEqual(result[0]['behavior_id'],'station_away')
        self.assertEqual(self.ledger.alarms(),[])
    def test_disputed_model_state_blocks_alarm_and_no_inside_disappearance_inferred(self):
        for t in (0,1):self.tick(t,[INSIDE])
        for t in (2,3,4):self.tick(t,[OUTSIDE])
        self.ledger.review({'observations':[{'person_id':'P001','state':'uncertain','station_id':'a','frame_id':'f_4','reason':'身份不清'}]},[{'frame_id':'f_4','time_sec':4}])
        self.assertEqual(self.ledger.alarms(),[]);self.assertEqual(self.person()['review_status'],'uncertain')
    def test_empty_station_uses_visible_model_count_and_work_schedule(self):
        for t in (0,1,2):self.tick(t,[])
        self.ledger.review({'stations':[{'station_id':'a','visible':False,'count':None,'frame_id':'f_2','reason':'遮挡'}]},[{'frame_id':'f_2','time_sec':2}])
        self.assertEqual(self.ledger.alarms(),[])
        self.ledger.review({'stations':[{'station_id':'a','visible':True,'count':0,'frame_id':'f_2','reason':'工位清晰为空'}]},[{'frame_id':'f_2','time_sec':2}])
        self.ledger.policy['work_hours']=[['09:00','18:00']]
        self.assertEqual(self.ledger.alarms(),[])  # File clock unspecified: no guessing.
        self.ledger.scene['start_clock']='09:00'
        self.assertEqual(self.ledger.alarms()[0]['behavior_id'],'station_empty')
    def test_initially_empty_station_does_not_invent_a_missing_person(self):
        for t in (0,1,2):self.tick(t,[])
        self.assertEqual(self.ledger.snapshot()['people'],[])
    def test_invalid_timestamps_regions_and_policy_rejected(self):
        for t in (True,float('nan'),-1):
            with self.assertRaises(ValueError):self.tick(t,[])
        self.tick(0,[])
        with self.assertRaises(ValueError):self.tick(0,[])
        for scene in ({'stations':[]},{**SCENE,'policy':{'sample_fps':100}},{**SCENE,'stations':[{**SCENE['stations'][0],'polygon':[[0,0],[1,1],[0,1],[1,0]]}]}):
            with self.assertRaises(ValueError):validate_scene(scene)
    def test_manual_association_preserves_totals_and_future_tracks(self):
        self.tick(0,[INSIDE]);self.tick(1,[INSIDE]);self.tick(2,[])
        self.tick(4,[{**INSIDE,'person_id':'P002'}]);self.tick(5,[{**INSIDE,'person_id':'P002'}])
        self.ledger.bind('P001','员工A','a');self.ledger.merge('P002','P001')
        self.tick(6,[{**INSIDE,'person_id':'P002'}])
        self.assertEqual(len(self.ledger.people),1)
        person=self.person();self.assertEqual(person['name'],'员工A');self.assertEqual(person['at_station_sec'],4)
        self.assertEqual(person['unknown_sec'],2);self.assertEqual(person['identity_status'],'manual')
    def test_concurrent_people_cannot_be_merged(self):
        self.tick(0,[INSIDE,{**INSIDE,'person_id':'P002'}])
        with self.assertRaises(ValueError):self.ledger.merge('P001','P002')
    def test_late_model_review_can_confirm_departure_after_person_returned(self):
        for t,tracks in [(0,[INSIDE]),(1,[INSIDE]),(2,[OUTSIDE]),(3,[OUTSIDE]),(4,[OUTSIDE]),(5,[INSIDE]),(6,[INSIDE])]:self.tick(t,tracks)
        self.ledger.review({'observations':[{'person_id':'P001','state':'outside','station_id':'a','frame_id':'f_3','reason':'历史画面中明确离开'}]},[{'frame_id':'f_3','time_sec':3}])
        alert=self.ledger.alarms()[0]
        self.assertEqual((alert['t_start'],alert['t_end'],alert['duration_sec']),(2,5,3))
    def test_unusable_picture_is_unknown_and_cannot_report_empty_station(self):
        self.tick(0,[INSIDE]);self.tick(1,[INSIDE])
        self.ledger.observe(2,[],'f_2',observable=False)
        self.ledger.observe(3,[],'f_3',observable=False)
        report=self.ledger.snapshot();self.assertIsNone(report['observed_people'])
        self.assertEqual(report['people'][0]['state'],'unknown');self.assertEqual(report['people'][0]['away_count'],0)
        self.assertEqual(self.ledger.alarms(),[])
    def test_gap_during_same_departure_does_not_invent_second_departure(self):
        for t,tracks in [(0,[INSIDE]),(1,[INSIDE]),(2,[OUTSIDE]),(3,[OUTSIDE]),(10,[OUTSIDE]),(11,[OUTSIDE])]:self.tick(t,tracks)
        report=self.person();self.assertEqual(report['away_count'],1)
        self.assertEqual(report['away_sec'],2);self.assertEqual(report['longest_away_sec'],2)
        self.assertEqual(report['unknown_sec'],7)
    def test_new_person_after_long_absence_requires_association(self):
        for t,tracks in [(0,[INSIDE]),(1,[INSIDE]),(2,[OUTSIDE]),(3,[OUTSIDE]),(4,[]),(5,[{**INSIDE,'person_id':'P002'}]),(6,[{**INSIDE,'person_id':'P002'}])]:self.tick(t,tracks)
        people=self.ledger.snapshot()['people'];old=next(p for p in people if p['person_id']=='P001');new=next(p for p in people if p['person_id']=='P002')
        self.assertEqual(new['identity_status'],'pending');self.assertIn('P001',new['pending_links'])
        self.assertEqual(old['state'],'unknown')
    def test_old_review_cannot_overwrite_newer_conflicting_state(self):
        for t in (0,1,2):self.tick(t,[INSIDE])
        latest={'person_id':'P001','station_id':'a','state':'uncertain','frame_id':'f_2','reason':'当前遮挡'}
        older={**latest,'state':'at_station','frame_id':'f_1','reason':'旧图清晰'}
        self.ledger.review({'observations':[latest]},[{'frame_id':'f_2','time_sec':2}])
        self.ledger.review({'observations':[older]},[{'frame_id':'f_1','time_sec':1}])
        self.assertEqual(self.person()['review_status'],'uncertain')
    def test_merge_after_capture_gap_clips_overlapping_unknown_history(self):
        for t,tracks in [(0,[INSIDE]),(1,[INSIDE]),(2,[]),(4,[{**INSIDE,'person_id':'P002'}]),(5,[{**INSIDE,'person_id':'P002'}]),(10,[{**INSIDE,'person_id':'P002'}]),(11,[{**INSIDE,'person_id':'P002'}])]:self.tick(t,tracks)
        self.ledger.merge('P002','P001')
        spans=sorted(self.ledger.snapshot(include_intervals=True)['intervals'],key=lambda i:i['start'])
        for a,b in zip(spans,spans[1:]):self.assertLessEqual(a['end'],b['start'])
        person=self.person();self.assertEqual(person['at_station_sec']+person['away_sec']+person['unknown_sec'],11)
    def test_break_time_is_excluded_from_alert_threshold_but_retained_in_totals(self):
        self.ledger.scene['start_clock']='09:00';self.ledger.policy.update(away_sec=70,work_hours=[['09:00','09:05']],break_hours=[['09:01','09:03']])
        for t in range(121):self.tick(t,[INSIDE] if t<2 else [OUTSIDE])
        def review(t):self.ledger.review({'observations':[{'person_id':'P001','station_id':'a','state':'outside','frame_id':f'f_{t}','reason':'明确在工位外'}]},[{'frame_id':f'f_{t}','time_sec':t}])
        review(120);self.assertEqual(self.ledger.alarms(),[])
        for t in range(121,193):self.tick(t,[OUTSIDE])
        review(192);alert=self.ledger.alarms()[0]
        self.assertEqual(alert['active_duration_sec'],70);self.assertEqual(self.person()['away_sec'],190)
    def test_finish_closes_person_and_region_timelines_at_last_observed_time(self):
        self.tick(0,[INSIDE]);self.tick(1,[INSIDE]);self.ledger.finish()
        self.assertTrue(all(i['end'] is not None for i in self.ledger.intervals+self.ledger.station_intervals))
        self.assertEqual(self.person()['at_station_sec'],1)


class ProfilesAndProtocolTests(unittest.TestCase):
    def test_presets_copy_version_edit_default_delete_and_task_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            registry=ProfileRegistry(Path(directory),Path(__file__).resolve().parents[1]/'config/behaviors.yaml')
            self.assertEqual(len(registry.list()['profiles']),3)
            original=registry.load('workstations');snapshot=copy.deepcopy(original)
            cloned=registry.save('office',{**original,'name':'办公室'});registry.set_default('office')
            self.assertEqual(registry.list()['default_id'],'office')
            edited=registry.save('office',{**cloned,'station_policy':{**cloned['station_policy'],'away_sec':120}})
            self.assertNotEqual(cloned['version'],edited['version']);self.assertEqual(snapshot['station_policy']['away_sec'],60)
            registry.delete('office');self.assertEqual(registry.list()['default_id'],'current')
            with self.assertRaises(ValueError):registry.delete('current')
    def test_storage_survives_new_store_and_reanalysis_rejects_old_version(self):
        with tempfile.TemporaryDirectory() as directory:
            store=StationStore(Path(directory));store.save('job','v1',SCENE,{'people':[],'time_sec':3})
            self.assertEqual(StationStore(Path(directory)).load('job')['time_sec'],3)
            store.save('job','v2',SCENE,{'people':[],'time_sec':4})
            with self.assertRaises(ValueError):store.load('job','v1')
            store.delete('job')
            with self.assertRaises(FileNotFoundError):store.load('job')
    def test_station_protocol_rejects_hallucinated_identity_frame_and_invalid_counts(self):
        body={'frames':[{'b64':'image'}],'evidence':[{'frame_id':'f_1','time_sec':1}], 'stations':SCENE['stations'],'people':[{'person_id':'P001'}]}
        request=validate_request(body)
        result={'observations':[{'person_id':'P001','station_id':'a','state':'at_station','frame_id':'f_1','reason':'坐在工位'}],
                'stations':[{'station_id':'a','visible':True,'count':1,'frame_id':'f_1','reason':'清晰一人'}]}
        self.assertEqual(validate_result(result,request)['stations'][0]['count'],1)
        for key,value in [('person_id','P999'),('frame_id','f_999'),('state','guess'),('person_id',[])]:
            wrong=copy.deepcopy(result);wrong['observations'][0][key]=value
            with self.assertRaises(ValueError):validate_result(wrong,request)
        wrong=copy.deepcopy(result);wrong['stations'][0]['count']=True
        with self.assertRaises(ValueError):validate_result(wrong,request)
    def test_tracker_reordering_short_occlusion_and_ambiguity(self):
        tracker=AnonymousTracker(max_age=3)
        first=tracker.update(0,[{'bbox':[.1,.1,.25,.9]},{'bbox':[.7,.1,.9,.9]}])
        second=tracker.update(1,[{'bbox':[.69,.1,.89,.9]},{'bbox':[.11,.1,.26,.9]}])
        self.assertEqual([p['person_id'] for p in second],[first[1]['person_id'],first[0]['person_id']])
        tracker.update(2,[])
        third=tracker.update(3,[{'bbox':[.12,.1,.27,.9]}])
        self.assertEqual(third[0]['person_id'],first[0]['person_id'])
        fourth=tracker.update(10,[{'bbox':[.12,.1,.27,.9]}])
        self.assertNotEqual(fourth[0]['person_id'],first[0]['person_id'])
    def test_database_delete_rolls_back_when_media_delete_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            store=StationStore(Path(directory));store.save('job','v1',SCENE,{'time_sec':3})
            with self.assertRaises(OSError):
                with store.deleting('job'):raise OSError('media locked')
            self.assertEqual(store.load('job')['time_sec'],3)
            with store.deleting('job'):pass
            with self.assertRaises(FileNotFoundError):store.load('job')
    def test_model_multiple_valid_frames_and_conflicting_duplicates_are_conservative(self):
        request={'evidence':[{'frame_id':f'f_{i}','time_sec':i} for i in range(3)],'people':[{'person_id':'P001'}],'stations':SCENE['stations']}
        observations=[{'person_id':'P001','station_id':'a','state':'at_station','frame_id':f'f_{i}','reason':'真实画面中一人'} for i in range(3)]
        check={'station_id':'a','visible':True,'count':1,'frame_id':'f_2','reason':'清晰一人'}
        output=validate_result({'observations':observations,'stations':[check]},request)
        self.assertEqual(len(output['observations']),3)
        output=validate_result({'observations':[observations[2],{**observations[2],'state':'outside'}],
                                'stations':[check,{**check,'count':0}]},request)
        self.assertEqual(output['observations'][0]['state'],'uncertain')
        self.assertFalse(output['stations'][0]['visible']);self.assertIsNone(output['stations'][0]['count'])


if __name__=='__main__':unittest.main()
