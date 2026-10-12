"""Opt-in CPU tracking / deployed Qwen protocol check, with bounded GPU ownership."""
from __future__ import annotations
import base64
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
from check_validation_cluster import configure_auth
from civility.config import load_settings,load_behaviors,behavior_payload,parse_behaviors
from webui.cluster_client import ClusterClient,ServiceSession,deploy_service_package
from webui.behavior_profiles import ProfileRegistry
from webui.person_tracking import PersonDetector,AnonymousTracker
from webui.workstation_runner import annotate,file_frames,spatial_evidence
from webui.workstations import StationLedger
from webui.station_questions import prepare_question,validate_numbers
from civility.video_qa import validate_answer

OUT=ROOT/'runs/_workstations_20261012'


def main():
    import cv2
    OUT.mkdir(parents=True,exist_ok=True);configure_auth();cfg=load_settings();cc=cfg['cluster']
    client=ClusterClient(cc['remote_workdir'],cc['remote_python'],cc['remote_model_root'])
    if client.list_my_jobs():raise RuntimeError('已有远端作业，不重复创建测试 GPU')
    registry=ProfileRegistry(OUT/'real-profiles',ROOT/'config/behaviors.yaml')
    profile=registry.load('workstations')
    # Whole-frame ROI validates occupancy mechanics on the user's existing clip.
    # This is NOT office footage or a workstation field-accuracy benchmark.
    scene={'stations':[{'id':'test_region','name':'测试画面区域','polygon':[[0,0],[1,0],[1,1],[0,1]],'expected_people':1}],
           'policy':{'confirm_sec':1,'away_sec':2,'empty_sec':2}}
    ledger=StationLedger(scene);detector=PersonDetector(ROOT/'data/person_models/object_detection_yolox_2022nov.onnx');tracker=AnonymousTracker()
    samples=[];latencies=[]
    frames=file_frames(ROOT.parent/'test/吸烟.mp4',2,lambda:False)
    try:
        for index,(at,image,_,__) in enumerate(frames):
            if at>6:break
            started=time.perf_counter();tracks=tracker.update(at,detector.detect(image));latencies.append(time.perf_counter()-started)
            fid=f'f_{index}';ledger.observe(at,tracks,fid)
            samples.append({'frame_id':fid,'time_sec':at,'sample_index':index,'image':image,'tracks':tracks})
    finally:frames.close()
    selected=[samples[0],samples[len(samples)//2],samples[-1]]
    people=[{k:p.get(k) for k in ('person_id','name','home_station','identity_status')} for p in ledger.snapshot()['people']][:6]
    images=[];evidence=[]
    for sample in selected:
        marked=annotate(sample['image'],scene['stations'],sample['tracks']);ok,data=cv2.imencode('.jpg',marked)
        if not ok:raise RuntimeError('JPEG encode failed')
        images.append(base64.b64encode(data.tobytes()).decode('ascii'))
        (OUT/f"real-review-{sample['frame_id']}.jpg").write_bytes(data.tobytes())
        evidence.append(spatial_evidence({'frame_id':sample['frame_id'],'timestamp':sample['time_sec'],'index':sample['sample_index'],'tracks':sample['tracks']},scene['stations'],{p['person_id'] for p in people}))
    results={'status':'deploying','cpu_samples':len(samples),'cpu_mean_sec':sum(latencies)/len(latencies),
             'source':'test/吸烟.mp4','limitation':'ROI is the whole frame, not an office accuracy benchmark'}
    session=ServiceSession(client,cc['remote_model'],port=18193,exclude=cc['exclude_nodes'],extra_sbatch='--time=00:12:00')
    def persist(**extra):
        results.update(extra);results['job_id']=session.jobid
        (OUT/'real-model.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
    try:
        deploy_service_package(client,ROOT,behavior_payload(load_behaviors()))
        session.start(wait_ready_sec=240,progress=lambda message:persist(status=message))
        health=session.health();persist(status='reviewing',health={k:health.get(k) for k in ('model','device','workstation_protocol','video_qa_protocol')})
        if health.get('workstation_protocol')!=1:raise RuntimeError('New workstation protocol not loaded')
        import requests
        payload={'frames':[{'b64':b} for b in images],'evidence':evidence,'stations':scene['stations'],'people':people,
                 'behavior_rules':profile['behaviors'],'max_new_tokens':1536}
        response=requests.post(session.base_url()+'/workstation-review',json=payload,timeout=120);response.raise_for_status()
        review=response.json();persist(review=review)
        if review.get('error'):raise RuntimeError(review['error'])
        from civility.station_review import validate_request,validate_result
        review['parsed']=validate_result(review['parsed'],validate_request(payload))
        if not any(s['visible'] and s['count']>=1 for s in review['parsed']['stations']):
            raise RuntimeError('模型未正确核对包含明显人员的整个画面区域')
        ledger.review(review['parsed'],evidence)
        report={**ledger.snapshot(include_intervals=True),'use_mock':False}
        facts,refs,context=prepare_question(report,'P001在测试区域累计待了多少秒？请引用实际画面，累计时间以统计为准。')
        by_id={s['frame_id']:s for s in samples};qa_images=[];qa_refs=[]
        for fid in refs:
            sample=by_id[fid];ok,data=cv2.imencode('.jpg',sample['image']);qa_images.append(base64.b64encode(data.tobytes()).decode('ascii'))
            qa_refs.append({k:sample[k] for k in ('frame_id','time_sec','sample_index')})
        qa=session.ask_video(qa_images,'P001在测试区域累计待了多少秒？累计时间以程序统计为准。',qa_refs,context)
        validate_numbers(validate_answer(qa['parsed'],qa_refs),facts,qa_refs);persist(qa=qa,statistics=facts)
        construction=registry.load('construction')
        ordinary=session.infer(images[:1],['t=0s'],max_new_tokens=768,with_alert=True,behaviors=behavior_payload(parse_behaviors(construction)))
        if ordinary.get('error'):raise RuntimeError(ordinary['error'])
        persist(status='passed',ordinary_profile_switch=ordinary)
        print('PASS CPU tracking, deployed Qwen workstation review, statistics QA and warm profile switch',flush=True)
    except Exception as exc:
        persist(status='failed',error=f'{type(exc).__name__}: {exc}');raise
    finally:
        job_id=session.jobid;released=session.stop(sweep_orphans=False);remaining=client.list_my_jobs()
        for _ in range(15):
            if not any(j['jobid']==job_id for j in remaining):break
            time.sleep(2);remaining=client.list_my_jobs()
        results['validation_job_id']=job_id
        persist(released_job_ids=released,remaining_allocations=remaining)
        print('Released validation allocation; remaining:',remaining,flush=True)


if __name__=='__main__':main()
