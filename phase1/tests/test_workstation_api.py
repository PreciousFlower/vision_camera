"""Real file/API workflow with a deterministic detector adapter and fake VLM."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import time
import types
import unittest
from unittest.mock import Mock,patch
import cv2
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from fastapi.testclient import TestClient
from webui import server as S


class RectangleDetector:
    def detect(self,image):
        mask=cv2.inRange(image,np.array([0,0,150],np.uint8),np.array([100,100,255],np.uint8))
        contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        h,w=image.shape[:2];results=[]
        for contour in contours:
            x,y,bw,bh=cv2.boundingRect(contour)
            if bw*bh>1000:results.append({'bbox':[x/w,y/h,(x+bw)/w,(y+bh)/h],'confidence':.99})
        return results


class WorkstationApiTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.saved=S.STATE.cfg,S.STATE.jobs,S.STATE.uploads,S.STATE.session
        S.STATE.cfg=copy.deepcopy(S.STATE.cfg)
        S.STATE.cfg['paths'].update({key:str(self.root/key) for key in ('data_dir','frames_dir','results_dir')})
        S.STATE.jobs,S.STATE.uploads={},{}
        self.session=types.SimpleNamespace(state='ready',started_at=None,model='deployed-fixture',status=lambda:{'state':'ready'},
                    ask_video=Mock(side_effect=self.answer),health=lambda **kw:{'ready':True,'workstation_protocol':1})
        S.STATE.session=self.session
        self.patches=[patch.object(S,'ensure_current_runtime'),patch.object(S,'runtime_status',return_value={'protocol':3,'restart_required':False}),
                      patch('webui.workstation_runner.detector_for',return_value=RectangleDetector())]
        for p in self.patches:p.start()
        self.client=TestClient(S.create_app());self.client.__enter__()
        self.video=self.root/'walking.mp4'
        writer=cv2.VideoWriter(str(self.video),cv2.VideoWriter_fourcc(*'mp4v'),10,(480,270))
        for i in range(140):
            t=i/10;x=100 if t<2 else min(320,100+40*(t-2)) if t<9 else max(100,320-40*(t-9))
            image=np.full((270,480,3),225,np.uint8);cv2.rectangle(image,(int(x),45),(int(x)+40,230),(0,0,240),-1)
            writer.write(image)
        writer.release()
        self.scene={'stations':[{'id':'a','name':'1号工位','polygon':[[0,0],[.5,0],[.5,1],[0,1]],'expected_people':1}],
                    'policy':{'confirm_sec':1,'away_sec':2,'empty_sec':2,'review_interval_sec':5}}
        self.job_id=None
    def tearDown(self):
        for job in S.STATE.jobs.values():
            job.stop_flag=True
            worker=getattr(job,'_detect_worker',None)
            if worker and worker.is_alive():worker.join(10)
        self.client.__exit__(None,None,None)
        for p in reversed(self.patches):p.stop()
        S.STATE.cfg,S.STATE.jobs,S.STATE.uploads,S.STATE.session=self.saved
        self.temp.cleanup()
    def answer(self,frames,question,evidence,context):
        facts=json.loads(context)['trusted_workstation_statistics'];person=facts['people'][0]
        return {'parsed':{'answer':f"{person['person_id']}累计离岗{person['away_sec']:.1f}秒，离岗{person['away_count']}次。",
                         'citations':[{'frame_id':evidence[0]['frame_id'],'description':'引用对应观测画面；累计时间来自程序统计'}],
                         'limitations':'匿名人员连续抽样统计，未知区间单列'},'model':'deployed-fixture'}
    def upload(self):
        self.job_id=self.client.post('/api/upload',files={'file':('walking.mp4',self.video.read_bytes(),'video/mp4')}).json()['job_id']
        return self.job_id
    def start(self):
        if not self.job_id:self.upload()
        response=self.client.post('/api/detect/'+self.job_id,json={'profile_id':'workstations','station_scene':self.scene,'use_mock':True})
        self.assertEqual(response.status_code,200,response.text)
        worker=S.STATE.jobs[self.job_id]._detect_worker;worker.join(15)
        self.assertFalse(worker.is_alive())
        self.assertEqual(S.STATE.jobs[self.job_id].state,'done',S.STATE.jobs[self.job_id].error)
        return self.client.get('/api/stations/'+self.job_id).json()
    def test_file_statistics_durations_and_actual_registered_evidence(self):
        report=self.start();self.assertEqual(len(report['people']),1)
        person=report['people'][0];self.assertEqual(person['away_count'],1)
        self.assertAlmostEqual(person['away_sec'],7,places=1)
        self.assertAlmostEqual(person['at_station_sec']+person['away_sec']+person['unknown_sec'],report['time_sec'],places=3)
        away=next(i for i in report['intervals'] if i['state']=='away')
        url=f"/api/evidence/{self.job_id}?window_id=interval_{away['interval_id']}&revision={report['analysis_revision']}"
        evidence=self.client.get(url);self.assertEqual(evidence.status_code,200,evidence.text)
        for frame in evidence.json()['frames']:
            image=self.client.get(frame['image_url']);self.assertEqual(image.status_code,200,image.text[:200]);self.assertTrue(image.content.startswith(b'\xff\xd8'))
        self.assertTrue(any(a['behavior_id']=='station_away' for a in report['alerts']))
        compact=self.client.get(f'/api/stations/{self.job_id}?compact=true').json()
        self.assertNotIn('intervals',compact);self.assertEqual(compact['people'],report['people'])
        regional=next(i for i in report['station_intervals'] if i['category']=='empty')
        evidence=self.client.get(f"/api/evidence/{self.job_id}?window_id=station_interval_{regional['interval_id']}&revision={report['analysis_revision']}")
        self.assertEqual(evidence.status_code,200,evidence.text)
        for frame in evidence.json()['frames']:self.assertEqual(self.client.get(frame['image_url']).status_code,200)
    def test_station_qa_uses_program_numbers_and_rejects_invented_times(self):
        report=self.start()
        response=self.client.post(f'/api/videos/{self.job_id}/ask',json={'question':'P001累计离岗多久，几次？'})
        self.assertEqual(response.status_code,200,response.text);self.assertIn('7.0秒',response.json()['answer'])
        self.assertEqual(response.json()['station_statistics']['people'][0]['away_count'],1)
        self.session.ask_video.side_effect=lambda frames,question,evidence,context:{'parsed':{'answer':'累计离岗999秒','citations':[{'frame_id':evidence[0]['frame_id'],'description':'测试'}]}}
        self.assertEqual(self.client.post(f'/api/videos/{self.job_id}/ask',json={'question':'P001累计离岗多久？'}).status_code,422)
    def test_profile_scene_saved_and_task_snapshot_survives_edit_restart_delete(self):
        self.upload();scope={'job_id':self.job_id}
        self.assertEqual(self.client.post('/api/station-scenes/save',json={**scope,'scene':self.scene}).status_code,200)
        self.assertEqual(self.client.post('/api/station-scenes/load',json=scope).json()['scene']['stations'][0]['id'],'a')
        preview=self.client.post('/api/station-scenes/preview',json=scope);self.assertEqual(preview.status_code,200)
        report=self.start();version=S.STATE.jobs[self.job_id].profile['version']
        profile=self.client.get('/api/profiles/workstations').json();profile['station_policy']['away_sec']=120
        self.assertEqual(self.client.put('/api/profiles/workstations',json=profile).status_code,200)
        self.assertEqual(S.STATE.jobs[self.job_id].profile['version'],version)
        S.STATE.jobs.clear();S.STATE.uploads.clear()
        historical=self.client.get('/api/stations/'+self.job_id).json();self.assertEqual(historical['people'][0]['away_sec'],report['people'][0]['away_sec'])
        bind=self.client.post(f'/api/stations/{self.job_id}/person',json={'analysis_revision':historical['analysis_revision'],'person_id':'P001','name':'员工A','station_id':'a'})
        self.assertEqual(bind.status_code,200,bind.text)
        self.assertEqual(self.client.delete('/api/videos/'+self.job_id).status_code,200)
        self.assertEqual(self.client.get('/api/stations/'+self.job_id).status_code,404)
    def test_invalid_scene_and_loop_rejected_without_starting_or_changing_old_job(self):
        self.upload();job=S.STATE.jobs[self.job_id]
        for body in ({'profile_id':'workstations','use_mock':True},{'profile_id':'workstations','use_mock':True,'station_scene':self.scene,'loop':True}):
            self.assertEqual(self.client.post('/api/detect/'+self.job_id,json=body).status_code,400)
            self.assertEqual(job.state,'uploaded');self.assertFalse(hasattr(job,'_detect_worker'));self.assertFalse(job.loop)
    def test_detector_initialization_failure_is_archived_and_emits_terminal_event(self):
        self.upload()
        with patch('webui.workstation_runner.detector_for',side_effect=FileNotFoundError('model missing')):
            response=self.client.post('/api/detect/'+self.job_id,json={'profile_id':'workstations','station_scene':self.scene,'use_mock':True})
            self.assertEqual(response.status_code,200);job=S.STATE.jobs[self.job_id];job._detect_worker.join(10)
        self.assertEqual(job.state,'error');self.assertIn('model missing',job.error)
        self.assertEqual(self.client.get('/api/stations/'+self.job_id).status_code,200)
        events=[]
        while not job.events.empty():events.append(job.events.get())
        self.assertTrue(any(e['type']=='done' for e in events))


if __name__=='__main__':unittest.main()
