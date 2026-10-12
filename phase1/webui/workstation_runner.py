"""Chronological CPU tracking with asynchronous reviews by the deployed VLM."""
from __future__ import annotations
import base64
import concurrent.futures
import copy
from collections import deque
import functools
import json
import math
from pathlib import Path
import time

from .person_tracking import PersonDetector, AnonymousTracker
from .workstations import StationLedger


@functools.lru_cache(maxsize=2)
def detector_for(path):
    return PersonDetector(Path(path))


def annotate(image, stations, tracks):
    import cv2
    import numpy as np
    result=image.copy();h,w=result.shape[:2]
    for station in stations:
        polygon=np.array([[min(w-1,int(x*w)),min(h-1,int(y*h))] for x,y in station['polygon']],dtype=np.int32)
        cv2.polylines(result,[polygon],True,(170,110,65),2)
        label=(max(4,min(w-100,int(polygon[0][0])+6)),max(20,min(h-6,int(polygon[0][1])+20)))
        cv2.putText(result,station['id'],label,cv2.FONT_HERSHEY_SIMPLEX,.6,(170,110,65),2)
    for track in tracks:
        x1,y1,x2,y2=track['bbox'];a,b=(int(x1*w),int(y1*h)),(int(x2*w),int(y2*h))
        cv2.rectangle(result,a,b,(60,150,60),2)
        cv2.putText(result,track['person_id'],(a[0],max(15,a[1])),cv2.FONT_HERSHEY_SIMPLEX,.6,(60,150,60),2)
    return result


def spatial_evidence(sample, stations, person_ids):
    """Location hints help the VLM locate candidates, never certify visible persons."""
    from .workstations import contains
    hints=[]
    for track in sample['tracks']:
        if track['person_id'] not in person_ids:continue
        bbox=track['bbox'];center=[(bbox[0]+bbox[2])/2,bbox[1]+.6*(bbox[3]-bbox[1])]
        hints.append({'person_id':track['person_id'],'bbox':bbox,'body_center':center,
                      'candidate_stations':[s['id'] for s in stations if contains(s['polygon'],center)]})
    return {'frame_id':sample['frame_id'],'time_sec':sample['timestamp'],'sample_index':sample['index'],'geometry_hint':hints}


def file_frames(source, fps, stopped):
    import cv2
    cap=cv2.VideoCapture(str(source))
    try:
        if not cap.isOpened(): raise ValueError('视频无法打开')
        source_fps=cap.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(source_fps) or source_fps<=0: raise ValueError('视频没有可靠帧率，无法进行工位计时')
        duration=cap.get(cv2.CAP_PROP_FRAME_COUNT)/source_fps
        next_sample=0.;index=0;previous=-1.;last=None
        while not stopped():
            ok,image=cap.read()
            if not ok: break
            timestamp=cap.get(cv2.CAP_PROP_POS_MSEC)/1000
            if timestamp<=previous: timestamp=index/source_fps
            previous=timestamp;index+=1;last=(timestamp,image)
            if timestamp+1e-6>=next_sample:
                yield timestamp,image,duration,None
                next_sample=timestamp+1/fps;last=None
        if last and not stopped(): yield last[0],last[1],duration,None
    finally: cap.release()


def live_frames(job, source, cfg, stopped):
    from .capture import open_source, StreamReader
    cap,kind,_,error=open_source(str(source))
    if cap is None: raise ValueError(error)
    reader=StreamReader(cap,kind,max_side=960,name=job.job_id[:8])
    job.reader=reader;reader.start()
    origin=None;next_sample=0.;last=time.monotonic()
    try:
        while not stopped():
            with reader.lock: history=list(reader.station_history)
            fresh=False
            for monotonic_at,wall_at,jpeg in history:
                if origin is None: origin=monotonic_at
                timestamp=monotonic_at-origin
                if timestamp+1e-6<next_sample: continue
                import cv2
                import numpy as np
                image=cv2.imdecode(np.frombuffer(jpeg,np.uint8),cv2.IMREAD_COLOR)
                if image is None: continue
                next_sample=timestamp+1/job.station_scene['policy']['sample_fps']
                fresh=True;last=time.monotonic()
                yield timestamp,image,None,wall_at-timestamp
            if reader.dead: raise ValueError('摄像头断流，已结束本次统计；末尾不可观察区间不计为离岗，请重新连接')
            if not fresh:
                if time.monotonic()-last>10: raise ValueError('摄像头超过10秒没有新画面，本次统计结束')
                time.sleep(.03)
    finally:
        reader.stop();cap.release();job.reader=None


def run_station_job(job, source, cfg, session, library, store, log):
    import cv2
    import numpy as np
    directory=Path(cfg['paths']['frames_dir'])/job.job_id
    detector=None;pool=None;future=None;package=None
    ledger=StationLedger(job.station_scene,enabled=[b['id'] for b in job.profile['behaviors'] if b.get('enabled')])
    job._station_ledger=ledger
    buffer=deque(maxlen=64);last_review=-1.;last_publish=-1.;last_persist=0.;sequence=0
    saved=set();model_status='等待部署的大模型核验';last_review_time=-1.;last_error=''
    previous=None;frames=None
    job._archive_source=None if job.is_live else Path(source)
    job._archived_windows=0;job._archive_enabled=True
    def save_frame(sample):
        if sample['index'] not in saved:
            directory.mkdir(parents=True,exist_ok=True)
            (directory/f"f_{sample['index']:05d}.jpg").write_bytes(sample['jpeg']);saved.add(sample['index'])
            library.register_frame(job.job_id,sample['index'],sample['timestamp'])
    def persist():
        report={**ledger.snapshot(include_intervals=True),'state':job.state,'model_status':model_status,
                'use_mock':job.use_mock,'filename':job.filename,'profile':job.profile,'model_error':last_error,'is_live':job.is_live}
        store.save(job.job_id,job._analysis_revision,job.station_scene,report)
        library.save(job)
    def publish():
        job._station_model_status=model_status;job._station_model_error=last_error
        job.events.put({'type':'station','data':{**ledger.snapshot(),'analysis_revision':job._analysis_revision,'model_status':model_status,'model_error':last_error,'use_mock':job.use_mock,'profile':job.profile,'scene':job.station_scene,'progress':job.progress}})
    def review_call(payload):
        if job.use_mock:
            latest=payload['samples'][-1];tracks=latest['tracks'];check=[]
            for s in payload['stations']:
                check.append({'station_id':s['id'],'visible':True,'count':sum(t.get('station_id')==s['id'] for t in tracks),
                              'frame_id':latest['frame_id'],'reason':'演示核验，非真实大模型判断'})
                count=check[-1]['count']
                older=next((sample for sample in reversed(payload['samples'][:-1]) if sum(t.get('station_id')==s['id'] for t in sample['tracks'])!=count),None)
                if older:check.append({'station_id':s['id'],'visible':True,'count':sum(t.get('station_id')==s['id'] for t in older['tracks']),
                                      'frame_id':older['frame_id'],'reason':'演示历史人数核验，非真实模型判断'})
            observations=[]
            for p in payload['people']:
                visible=next((t for t in tracks if t['person_id']==p['person_id']),None)
                home=p.get('home_station');sid=visible.get('station_id') if visible else None
                state='at_station' if visible and sid and (not home or sid==home) else 'outside' if visible and home else 'uncertain'
                observations.append({'person_id':p['person_id'],'station_id':sid or home,'state':state,
                                     'frame_id':latest['frame_id'],'reason':'演示核验，非真实大模型判断'})
                if home:
                    older=next((sample for sample in reversed(payload['samples'][:-1]) if any(t['person_id']==p['person_id'] and t.get('station_id')!=home for t in sample['tracks'])),None)
                    if older: observations.append({'person_id':p['person_id'],'station_id':home,'state':'outside','frame_id':older['frame_id'],'reason':'演示历史离岗核验，非真实模型判断'})
            return {'parsed':{'scene':'工位演示统计','stations':check,'observations':observations},'raw':'演示数据','model':'mock'}
        return session.review_workstations(payload['images'],payload['evidence'],payload['stations'],payload['people'],job.profile)
    def submit(final=False):
        nonlocal future,package,last_review
        if not buffer or future is not None: return
        samples=list(buffer)
        if len(samples)>8:
            # Endpoints and chronological context; latest state always included.
            samples=[samples[round(i*(len(samples)-1)/7)] for i in range(8)]
        snapshot=ledger.snapshot();stations=job.station_scene['stations']
        # Cycle all configured stations / people through bounded model requests.
        batch_no=job.windows_seen
        chunks=[stations[i:i+4] for i in range(0,len(stations),4)]
        subset=chunks[batch_no%len(chunks)]
        people=snapshot['people'];visible_ids={t['person_id'] for t in samples[-1]['tracks']}
        needs_history={i['person_id'] for i in ledger.intervals if i['state']=='away' and i.get('review_status')!='confirmed'}
        people=sorted(people,key=lambda p:(p['person_id'] not in needs_history,p['review_status']=='confirmed',p['person_id'] not in visible_ids))[:6]
        images=[];evidence=[]
        person_ids={p['person_id'] for p in people}
        for sample in samples:
            save_frame(sample)
            image=cv2.imdecode(np.frombuffer(sample['jpeg'],np.uint8),cv2.IMREAD_COLOR)
            marked=annotate(image,subset,[t for t in sample['tracks'] if t['person_id'] in person_ids]);ok,encoded=cv2.imencode('.jpg',marked)
            if not ok: raise ValueError('工位复核画面编码失败')
            images.append(base64.b64encode(encoded.tobytes()).decode('ascii'))
            evidence.append(spatial_evidence(sample,subset,person_ids))
        package={'images':images,'evidence':evidence,'stations':subset,'people':[{k:p.get(k) for k in ('person_id','name','home_station','identity_status')} for p in people],
                 'samples':samples,'started':time.perf_counter()}
        future=pool.submit(review_call,package);last_review=ledger.time
    def collect():
        nonlocal future,package,model_status,last_review_time,last_error
        if future is None or not future.done(): return
        selected=package
        try:
            result=future.result()
            from civility.station_review import validate_result
            parsed=validate_result(result.get('parsed'),{'evidence':selected['evidence'],'people':selected['people'],'stations':selected['stations']})
            ledger.review(parsed,selected['evidence']);last_review_time=selected['evidence'][-1]['time_sec']
            model_status='演示核验（模拟结论）' if job.use_mock else '已由部署的大模型核验'
            last_error=''
            scene=parsed['scene']
        except Exception as exc:
            parsed={'observations':[],'stations':[]};scene='本次工位复核失败，统计保留；未经模型确认不发告警'
            last_error=f'{type(exc).__name__}: {exc}';model_status='大模型核验暂不可用，待复核'
            log(f'[{job.job_id[:8]}] 工位复核失败：{last_error}')
        finally: future=None;package=None
        samples=selected['samples'];wid=f'{job.job_id}__w{job.windows_seen:05d}'
        times={e['frame_id']:e['time_sec'] for e in selected['evidence']}
        latest_people={}
        for observation in sorted(parsed['observations'],key=lambda o:times[o['frame_id']]): latest_people[observation['person_id']]=observation
        row={'window_id':wid,'t_start':samples[0]['timestamp'],'t_end':samples[-1]['timestamp'],
             'frames':len(samples),'frame0':samples[0]['index'],'frame_indices':[s['index'] for s in samples],
             'frame_labels':[f"t={s['timestamp']:.3f}s" for s in samples],'frame_times':[s['timestamp'] for s in samples],
             'scene':scene,'people':[{'where':o.get('station_id') or '工位外','appearance':o['person_id'],'action':o['state'],'changed':o['reason']} for o in latest_people.values()],
             'station_review':parsed,'violations':{},'risks':[],'latency':time.perf_counter()-selected['started'],
             'error':last_error or None,'tokens_out':result.get('tokens_out',0) if not last_error else 0}
        job.windows_seen+=1;job.windows_done=job.windows_seen;job.window_results.append(row)
        job.window_results=job.window_results[-200:];library.record(job,row);job._archived_windows+=1
        job.events.put({'type':'window',**row,'index':job.windows_seen,'cumulative':job.windows_seen,'hits':[],'progress':job.progress})
        fresh=ledger.alarms()
        for alert in fresh:
            anchor=('interval_' if alert.get('person_id') else 'station_interval_')+alert['interval_id']
            alert['window_ids']=[anchor,wid]
        if fresh:
            job.alerts.extend(fresh);job.alerts=job.alerts[-500:]
            job.events.put({'type':'alerts','alerts':job.alerts});log(f'[{job.job_id[:8]}] 工位告警 {len(fresh)} 条')
        persist();publish()
    try:
        job.state='running';job.message='工位人员连续统计中，大模型异步核验'
        library.begin(job);directory.mkdir(parents=True,exist_ok=True)
        detector=detector_for(str(Path(cfg['paths']['data_dir'])/'person_models/object_detection_yolox_2022nov.onnx'))
        tracker=AnonymousTracker(max_age=ledger.policy['occlusion_sec'])
        pool=concurrent.futures.ThreadPoolExecutor(max_workers=1,thread_name_prefix='station-review')
        frames=live_frames(job,source,cfg,lambda:job.stop_flag) if job.is_live and job.source_kind!='file' else file_frames(source,ledger.policy['sample_fps'],lambda:job.stop_flag)
        previous=None
        for timestamp,image,duration,wall_origin in frames:
            if job.stop_flag: break
            if wall_origin is not None: ledger.wall_origin=wall_origin
            gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            observable=10<float(gray.mean())<245 and float(cv2.Laplacian(gray,cv2.CV_64F).var())>=2
            tracks=tracker.update(timestamp,detector.detect(image) if observable else [])
            before={p['person_id']:(p['interval']['interval_id'],p['interval']['state']) for p in ledger.people.values()}
            before_stations={sid:s['interval_id'] for sid,s in ledger.station_since.items()}
            frame_id=f'f_{sequence}';sequence+=1
            ledger.observe(timestamp,tracks,frame_id,observable=observable)
            if max(image.shape[:2])>960:
                ratio=960/max(image.shape[:2]);image=cv2.resize(image,(int(image.shape[1]*ratio),int(image.shape[0]*ratio)))
            ok,encoded=cv2.imencode('.jpg',image,[cv2.IMWRITE_JPEG_QUALITY,82])
            if not ok: raise ValueError('证据画面编码失败')
            sample={'timestamp':timestamp,'index':sequence-1,'frame_id':frame_id,'jpeg':encoded.tobytes(),'tracks':copy.deepcopy(tracks)}
            buffer.append(sample)
            job._station_buffer=buffer
            transition=any(before.get(pid)!=(p['interval']['interval_id'],p['interval']['state']) for pid,p in ledger.people.items()) or any(before_stations.get(sid)!=s['interval_id'] for sid,s in ledger.station_since.items())
            if transition:
                save_frame(sample)
                if previous: save_frame(previous)
            previous=sample
            collect()
            if future is None and (last_review<0 or timestamp-last_review>=ledger.policy['review_interval_sec'] or (transition and timestamp-last_review>=3)): submit()
            if timestamp-last_publish>=1:
                job.message=f'工位统计 {timestamp:.1f} 秒 · '+(f'{sum(ledger.counts.values())} 人' if observable else '画面不可观察，未知计时')+f' · {model_status}'
                if duration: job.progress=min(.98,timestamp/duration)
                publish();last_publish=timestamp
            if time.monotonic()-last_persist>=2: persist();last_persist=time.monotonic()
        if not job.stop_flag:
            # Finish the existing review and one final view of the complete tail.
            for final_pass in range(2):
                if future:
                    while not future.done() and not job.stop_flag: time.sleep(.05)
                    if job.stop_flag: break
                    collect()
                if ledger.time is not None and last_review_time<ledger.time and final_pass==0: submit(final=True)
        job.state='stopped' if job.stop_flag else 'done';job.progress=1.
        job.message=f'工位统计已{"停止" if job.stop_flag else "完成"} · {ledger.time or 0:.1f} 秒 · {len(ledger.people)} 个匿名编号'
    except Exception as exc:
        job.state='error';job.error=f'{type(exc).__name__}: {exc}';job.message=job.error
        log(f'[{job.job_id[:8]}] 工位统计失败：{job.error}')
    finally:
        if future: future.cancel()
        if pool: pool.shutdown(wait=False,cancel_futures=True)
        try:
            if frames is not None: frames.close()
        except Exception as exc:
            job.error=(job.error+'；' if job.error else '')+f'采集关闭失败：{exc}'
        try:
            ledger.finish()
            if previous is not None: save_frame(previous)
            persist()
        except Exception as exc:
            job.error=(job.error+'；' if job.error else '')+f'统计保存失败：{exc}'
        publish();job.events.put({'type':'done','summary':job.summary()})
