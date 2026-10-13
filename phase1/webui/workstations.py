"""Workstation geometry, time ledger and evidence-gated alarms.

The ledger consumes chronological capture timestamps, never inference duration.
Missing observations stay unknown. Anonymous IDs are not employee identities.
"""
from __future__ import annotations
import copy
import datetime as dt
import json
import math
import re
import sqlite3
import threading
import uuid
from contextlib import closing, contextmanager
from pathlib import Path

POLICY_DEFAULTS = {'sample_fps': 2., 'confirm_sec': 3., 'max_gap_sec': 2.5,
                   'occlusion_sec': 3., 'review_interval_sec': 10., 'away_sec': 60.,
                   'empty_sec': 60., 'work_hours': [], 'break_hours': []}
POLICY_DEFAULTS['alarm_mode']='immediate'
UNKNOWN_REASONS={'target_missing':'目标暂时丢失，可能为遮挡或漏检，尚未确认离岗',
                 'capture_gap':'采样时间断档，无法确认这段时间的位置',
                 'unobservable':'画面不可观察（断流、黑屏、过曝或模糊）',
                 'confirming':'首次出现或状态变化尚在防抖确认',
                 'identity_pending':'返回人员身份待关联，不能替旧编号继续计时',
                 'model_occluded':'大模型判断该区域被遮挡或不可观察'}
LIMITS = {'sample_fps': (.5, 5), 'confirm_sec': (0, 30), 'max_gap_sec': (.5, 30),
          'occlusion_sec': (0, 30), 'review_interval_sec': (5, 300), 'away_sec': (1, 86400), 'empty_sec': (1, 86400)}


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def validate_policy(raw):
    if not isinstance(raw, dict): raise ValueError('工位策略必须是对象')
    result = copy.deepcopy(POLICY_DEFAULTS)
    mode=raw.get('alarm_mode',result['alarm_mode'])
    if mode not in ('immediate','confirmed'):raise ValueError('告警模式须为immediate（阈值即提示）或confirmed（模型确认后）')
    result['alarm_mode']=mode
    for key, bounds in LIMITS.items():
        value = raw.get(key, result[key])
        if not finite(value) or not bounds[0] <= value <= bounds[1]: raise ValueError(f'{key} 需在 {bounds[0]}–{bounds[1]} 范围内')
        result[key] = float(value)
    if result['max_gap_sec'] < 1/result['sample_fps']: raise ValueError('允许采样空档不能短于正常采样间隔')
    for key in ('work_hours', 'break_hours'):
        hours = raw.get(key, [])
        if not isinstance(hours, list) or len(hours)>8: raise ValueError('工作/休息时间最多配置8段')
        for pair in hours:
            if not isinstance(pair, list) or len(pair)!=2 or any(not isinstance(x,str) or not re.fullmatch(r'(?:[01]\d|2[0-3]):[0-5]\d', x) for x in pair):
                raise ValueError('时间段使用 ["09:00", "18:00"] 格式')
            if pair[0]==pair[1]: raise ValueError('时间段起止不能相同')
        result[key] = copy.deepcopy(hours)
    return result


def validate_scene(raw):
    if not isinstance(raw, dict): raise ValueError('工位配置必须是对象')
    stations = raw.get('stations')
    if not isinstance(stations, list) or not 1<=len(stations)<=20: raise ValueError('请绘制1至20个工位区域')
    result, seen = [], set()
    for item in stations:
        if not isinstance(item, dict): raise ValueError('工位定义无效')
        sid, name, polygon = item.get('id'), item.get('name'), item.get('polygon')
        if not isinstance(sid,str) or not re.fullmatch(r'[a-zA-Z0-9_-]{1,32}',sid) or sid in seen: raise ValueError('工位编号重复或无效')
        if not isinstance(name,str) or not name.strip() or len(name)>80: raise ValueError('工位名称须为1至80字')
        if not isinstance(polygon,list) or not 3<=len(polygon)<=20: raise ValueError('工位区域需要3至20个点')
        if any(not isinstance(p,list) or len(p)!=2 or any(not finite(v) or not 0<=v<=1 for v in p) for p in polygon): raise ValueError('工位点坐标必须在0至1之间')
        area = abs(sum(polygon[i][0]*polygon[(i+1)%len(polygon)][1]-polygon[(i+1)%len(polygon)][0]*polygon[i][1] for i in range(len(polygon))))/2
        if area<.0001: raise ValueError('工位区域过小或点重叠')
        def cross(a,b,c): return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
        for i in range(len(polygon)):
            for j in range(i+2,len(polygon)):
                if i==0 and j==len(polygon)-1: continue
                a,b,c,d = polygon[i],polygon[(i+1)%len(polygon)],polygon[j],polygon[(j+1)%len(polygon)]
                if cross(a,b,c)*cross(a,b,d)<0 and cross(c,d,a)*cross(c,d,b)<0: raise ValueError('工位多边形不能自相交')
        expected = item.get('expected_people',1)
        if type(expected) is not int or not 0<=expected<=30: raise ValueError('预期人数需为0至30的整数')
        seen.add(sid); result.append({'id':sid,'name':name.strip(),'polygon':copy.deepcopy(polygon),'expected_people':expected})
    start_clock = raw.get('start_clock','')
    if not isinstance(start_clock,str) or (start_clock and not re.fullmatch(r'(?:[01]\d|2[0-3]):[0-5]\d',start_clock)): raise ValueError('视频起始钟点使用HH:MM格式')
    return {'stations':result,'policy':validate_policy(raw.get('policy') or {}),'start_clock':start_clock}


def contains(polygon, point):
    x,y = point; inside = False
    for i,a in enumerate(polygon):
        b = polygon[(i+1)%len(polygon)]
        if (a[1]>y)!=(b[1]>y) and x < (b[0]-a[0])*(y-a[1])/(b[1]-a[1])+a[0]: inside = not inside
    return inside


class StationLedger:
    def __init__(self, scene, *, enabled=None, wall_origin=None):
        self.scene = validate_scene(scene); self.policy = self.scene['policy']
        self.people, self.intervals, self.station_checks, self.aliases = {}, [], {}, {}
        self.enabled = set(enabled if enabled is not None else ('station_away','station_empty','station_over_capacity'))
        self.time = None; self.wall_origin = wall_origin; self.last_frames=[]; self.counts={s['id']:0 for s in self.scene['stations']}
        self.station_since = {}; self.station_intervals=[]; self.alerted=set(); self.alerts=[]; self.gaps=[]
        self.lock = threading.RLock()
        self.max_observed_people=0

    def _interval(self, person, timestamp, state, station_id=None, frame_id=None, unknown_reason_code=None):
        old = person.get('interval')
        if old and old['state']==state and old.get('station_id')==station_id and old.get('unknown_reason_code')==unknown_reason_code:
            return old
        if old: old['end']=timestamp
        if old and old['start']==timestamp and old['end']==timestamp:self.intervals.remove(old)
        item = {'interval_id':uuid.uuid4().hex,'person_id':person['person_id'],'start':timestamp,'end':None,
                'state':state,'station_id':station_id,'frame_start':frame_id,'frame_end':frame_id,
                'frame_start_time':timestamp,'frame_end_time':timestamp,'review_status':'pending'}
        if state=='unknown':item.update(unknown_reason_code=unknown_reason_code or 'confirming',unknown_reason=UNKNOWN_REASONS.get(unknown_reason_code or 'confirming'))
        self.intervals.append(item); person['interval']=item
        return item

    def station_for(self, bbox):
        point=((bbox[0]+bbox[2])/2, bbox[1]+.6*(bbox[3]-bbox[1]))
        found=[s for s in self.scene['stations'] if contains(s['polygon'],point)]
        # Unique assignment even when drawn regions overlap.
        return min(found,key=lambda s:abs(sum(s['polygon'][i][0]*s['polygon'][(i+1)%len(s['polygon'])][1]-s['polygon'][(i+1)%len(s['polygon'])][0]*s['polygon'][i][1] for i in range(len(s['polygon'])))))['id'] if found else None

    def observe(self, timestamp, tracks, frame_id, observable=True):
        if not finite(timestamp) or timestamp<0: raise ValueError('采集时间无效')
        with self.lock:
            if self.time is not None and timestamp<=self.time: raise ValueError('采集时间必须递增')
            if len(self.intervals)>=100000: raise ValueError('本次统计达到容量上限，已保留记录，请开启新的监测任务')
            if self.time is not None and timestamp-self.time>self.policy['max_gap_sec']:
                self.gaps.append({'start':self.time,'end':timestamp})
                for person in self.people.values():
                    self._interval(person,self.time,'unknown',person.get('home_station'),self.last_frames[-1]['frame_id'] if self.last_frames else None,'capture_gap')
                    person['pending']=None
                    person['missing_since']=timestamp;person['exit_observed']=False
                for span in self.station_since.values(): span['end']=self.time
                self.station_since.clear(); self.station_checks.clear()
            self.time=timestamp
            if not observable:
                for p in self.people.values():
                    self._interval(p,timestamp,'unknown',p.get('home_station'),frame_id,'unobservable')
                    p['visible']=False;p['pending']=None
                    p['missing_since']=timestamp;p['exit_observed']=False
                for span in self.station_since.values():span['end']=timestamp
                self.station_since.clear();self.station_checks.clear();self.counts={s['id']:0 for s in self.scene['stations']}
                self.view_observable=False
                return
            self.view_observable=True
            for track in tracks:
                for _ in range(len(self.aliases)):
                    if track['person_id'] not in self.aliases: break
                    track['person_id']=self.aliases[track['person_id']]
            visible={track['person_id']:track for track in tracks}
            self.counts={s['id']:0 for s in self.scene['stations']}
            for track in tracks:
                sid=self.station_for(track['bbox'])
                track['station_id']=sid
                if sid: self.counts[sid]+=1
            self.max_observed_people=max(self.max_observed_people,sum(self.counts.values()))
            for pid in set(self.people)|set(visible):
                track=visible.get(pid); person=self.people.get(pid)
                if person is None:
                    if not track.get('station_id'): continue
                    if len(self.people)>=500: raise ValueError('本次统计达到人员容量上限，已保留记录，请开启新的监测任务')
                    person={'person_id':pid,'name':pid,'home_station':None,'first_seen':timestamp,'pending':None,
                            'identity_status':track.get('identity_status','anonymous'),'pending_links':track.get('pending_links',[])}
                    candidates=[other['person_id'] for other in self.people.values() if other.get('home_station')==track.get('station_id') and other['person_id'] not in visible]
                    if candidates: person.update(identity_status='pending',pending_links=candidates)
                    self.people[pid]=person
                    self._interval(person,timestamp,'unknown',frame_id=frame_id)
                if track:
                    person['missing_since']=None
                    person.update(bbox=list(track['bbox']),last_seen=timestamp,visible=True,
                                  identity_status=person['identity_status'] if person.get('identity_status') in ('manual','pending') else track.get('identity_status','anonymous'),
                                  pending_links=[] if person.get('identity_status')=='manual' else person.get('pending_links') or track.get('pending_links',[]))
                    sid=track['station_id']
                    candidate='at_station' if sid else ('away' if person['home_station'] else 'unknown')
                    # Leaving another workstation also means leaving the assigned home.
                    if person['home_station'] and sid!=person['home_station']: candidate='away'
                    person['observed_station']=sid
                    if candidate=='away': sid=person['home_station']
                    person['exit_observed']=candidate=='away'
                    if track.get('departure_evidence'):person['departure_frame']=frame_id;person['departure_time']=timestamp
                else:
                    person['visible']=False; sid=person.get('home_station')
                    if person.get('missing_since') is None:person['missing_since']=timestamp
                    # A person already seen outside stays away while the view is
                    # healthy; disappearance from inside is unknown, not departure.
                    candidate='away' if person.get('known_left') or person['interval']['state']=='away' or person.get('exit_observed') else 'unknown'
                    home_people=[p for p in self.people.values() if p.get('home_station')==sid]
                    single_home=bool(sid) and len(home_people)==1 and person.get('identity_status')!='pending'
                    check=self.station_checks.get(sid,{})
                    model_occluded=check.get('visible') is False and timestamp-check.get('time_sec',-1)<=max(30,self.policy['review_interval_sec']*3)
                    grace=min(self.policy['occlusion_sec'],self.policy['away_sec']) if self.policy['alarm_mode']=='immediate' else self.policy['occlusion_sec']
                    if single_home and self.counts.get(sid)==0 and timestamp-person['missing_since']>=grace and not model_occluded:
                        candidate='away';person['absence_inferred']=True
                        if person['interval']['state']=='unknown' and person['interval'].get('unknown_reason_code')=='target_missing':
                            person['interval']['state']='away';person['interval'].pop('unknown_reason',None);person['interval'].pop('unknown_reason_code',None)
                            person['interval']['absence_inferred']=True
                            person['known_left']=True
                            if not person.get('away_episode'):person['away_episode']=uuid.uuid4().hex
                            person['interval']['away_episode']=person['away_episode'];person['pending']=None
                    if model_occluded:candidate='unknown'
                    if any(t.get('station_id')==sid and t['person_id']!=pid and self.people.get(t['person_id'],{}).get('identity_status')=='pending' for t in tracks): candidate='unknown'
                interval=person['interval']; interval['frame_end']=frame_id;interval['frame_end_time']=timestamp
                target=(candidate,sid)
                current=(interval['state'],interval['station_id'])
                if candidate=='unknown':
                    code='identity_pending' if person.get('identity_status')=='pending' or any(t.get('station_id')==sid and t['person_id']!=pid for t in tracks) else 'model_occluded' if not track and model_occluded else 'target_missing'
                    self._interval(person,timestamp,'unknown',sid,frame_id,code)
                    person['pending']=None
                elif current==target:
                    person['pending']=None
                else:
                    pending=person.get('pending')
                    if not pending or pending['target']!=target:
                        self._interval(person,timestamp,'unknown',sid,frame_id,'confirming')
                        person['pending']={'target':target,'since':timestamp}
                        pending=person['pending']
                    confirmation=min(self.policy['confirm_sec'],self.policy['away_sec']) if candidate=='away' and self.policy['alarm_mode']=='immediate' else self.policy['confirm_sec']
                    if timestamp-pending['since']>=confirmation:
                        person['interval']['state']=candidate
                        person['interval']['station_id']=sid
                        if candidate=='at_station' and not person['home_station']: person['home_station']=sid
                        if candidate=='at_station': person['known_left']=False;person['away_episode']=None;person['absence_inferred']=False
                        if candidate=='away':
                            person['known_left']=True
                            if not person.get('away_episode'):person['away_episode']=uuid.uuid4().hex
                            person['interval']['away_episode']=person['away_episode']
                        person['pending']=None
                        person['interval'].pop('unknown_reason',None);person['interval'].pop('unknown_reason_code',None)
            for sid,count in self.counts.items():
                expected=next(s['expected_people'] for s in self.scene['stations'] if s['id']==sid)
                category='empty' if count<expected else 'over' if count>expected else 'normal'
                previous=self.station_since.get(sid)
                if previous: previous.update(frame_end=frame_id,frame_end_time=timestamp)
                if not previous or previous['category']!=category or previous['count']!=count:
                    if previous: previous['end']=timestamp
                    span={'interval_id':uuid.uuid4().hex,'station_id':sid,'category':category,'count':count,'since':timestamp,'end':None,
                          'frame_id':frame_id,'frame_end':frame_id,'frame_end_time':timestamp}
                    self.station_since[sid]=span;self.station_intervals.append(span)
            self.last_frames.append({'frame_id':frame_id,'time_sec':timestamp})
            self.last_frames=self.last_frames[-24:]

    def review(self, result, evidence):
        with self.lock:
            by_id={f['frame_id']:f for f in evidence}
            for item in result.get('observations',[]):
                pid=item['person_id']
                for _ in range(len(self.aliases)):
                    if pid not in self.aliases:break
                    pid=self.aliases[pid]
                person=self.people.get(pid); frame=by_id.get(item['frame_id'])
                if not person or not frame: continue
                at=frame['time_sec']
                interval=next((x for x in reversed(self.intervals) if x['person_id']==pid and x['start']<=at and (x['end'] is None or at<x['end'])),None)
                if interval is None: continue
                if at<interval.get('review_time',-1): continue
                expected='at_station' if interval['state']=='at_station' else 'outside' if interval['state']=='away' else 'uncertain'
                confirmed=item['state']==expected and (expected!='at_station' or item.get('station_id')==interval.get('station_id'))
                rejected=interval['state']=='away' and item['state']=='at_station' and item.get('station_id')==interval.get('station_id')
                interval.update(review_status='confirmed' if confirmed and expected!='uncertain' else 'rejected' if rejected else 'uncertain',
                                model_reason=item['reason'],review_time=at,review_frame=item['frame_id'])
                # Contradictory geometry cannot become a confirmed absence alarm.
            for check in result.get('stations',[]):
                if check['frame_id'] in by_id:
                    checked={**check,'time_sec':by_id[check['frame_id']]['time_sec']}
                    if checked['time_sec']>=self.station_checks.get(check['station_id'],{}).get('time_sec',-1): self.station_checks[check['station_id']]=checked
                    span=next((s for s in reversed(self.station_intervals) if s['station_id']==check['station_id'] and s['since']<=checked['time_sec'] and (s['end'] is None or checked['time_sec']<s['end'])),None)
                    if span is not None: span['ai_check']=checked
                    residents=[p for p in self.people.values() if p.get('home_station')==check['station_id'] and p.get('identity_status')!='pending']
                    if len(residents)==1:
                        person=residents[0];at=checked['time_sec']
                        interval=next((i for i in reversed(self.intervals) if i['person_id']==person['person_id'] and i['start']<=at and (i['end'] is None or at<i['end'])),None)
                        if interval is not None and interval.get('absence_inferred') and interval['state']=='away' and at>=interval.get('review_time',-1):
                            status='confirmed' if checked['visible'] and checked['count']==0 else 'rejected' if checked['visible'] and checked['count']>0 else 'uncertain'
                            interval.update(review_status=status,review_frame=checked['frame_id'],review_time=at,
                                            model_reason='单人工位人数核对：'+checked['reason']+'；仅确认当前区域缺席，不补写未拍到的离开动作')

    def _work_time(self, timestamp=None):
        at=self.time if timestamp is None else timestamp
        if not self.policy['work_hours'] and not self.policy['break_hours']: return True
        if self.wall_origin is not None:
            local=dt.datetime.fromtimestamp(self.wall_origin+at,dt.timezone(dt.timedelta(hours=8)))
            minutes=local.hour*60+local.minute
        elif self.scene['start_clock']:
            h,m=map(int,self.scene['start_clock'].split(':')); minutes=(h*60+m+at/60)%1440
        else: return False
        def included(pair):
            a,b=([int(v) for v in x.split(':')] for x in pair)
            start,end=a[0]*60+a[1],b[0]*60+b[1]
            return start<=minutes<end if start<end else minutes>=start or minutes<end
        return (not self.policy['work_hours'] or any(included(p) for p in self.policy['work_hours'])) and not any(included(p) for p in self.policy['break_hours'])

    def alarms(self):
        with self.lock:
            if self.time is None: return []
            fresh=[]
            def add(key,bid,name,since,frame,reason,person=None,station=None,end=None,interval_id=None,status='confirmed',threshold=0):
                if bid not in self.enabled:return
                until=self.time if end is None else end
                lo,hi=since,until
                for _ in range(40):
                    mid=(lo+hi)/2
                    if self._active_seconds(since,mid)>=threshold:hi=mid
                    else:lo=mid
                deadline=round(hi,6)
                previous=next((a for a in self.alerts if a.get('alarm_id')==key),None)
                if previous and previous.get('review_status','confirmed')==status:return
                if status=='rejected' and previous is None:return
                if self.policy['alarm_mode']=='confirmed' and status!='confirmed' and previous is None:return
                alert={'behavior_id':bid,'behavior_name':name,'severity':'medium','t_start':since,'t_end':until,
                       'reason':reason,'evidence_frame':frame,'person_id':person,'station_id':station,
                       'source':'workstation','duration_sec':until-since,'active_duration_sec':self._active_seconds(since,until),
                       'interval_id':interval_id or key,'window_ids':[],'alarm_id':key,'review_status':status,
                       'provisional':status=='pending','threshold_time_sec':deadline,'trigger_time_sec':self.time,
                       'verification_delay_sec':max(0,self.time-deadline) if status=='confirmed' else None}
                if previous:
                    alert['trigger_time_sec']=previous['trigger_time_sec'];alert['window_ids']=previous.get('window_ids',[])
                    previous.update(alert);fresh.append(previous)
                else:self.alerted.add(key);self.alerts.append(alert);fresh.append(alert)
            for interval in self.intervals:
                pid=interval['person_id'];person=self.people[pid]
                until=interval['end'] if interval['end'] is not None else self.time
                if (interval['state']=='away' and person.get('identity_status')!='pending'
                        and self._active_seconds(interval['start'],until)>=self.policy['away_sec']
                        and self._work_time(until if interval['end'] is None else max(interval['start'],until-1e-6))):
                    status=interval.get('review_status','pending');status=status if status in ('confirmed','rejected') else 'pending'
                    if status=='confirmed' and interval['end'] is None and self.time-interval.get('review_time',-1)>max(30,self.policy['review_interval_sec']*3):status='pending'
                    add(interval['interval_id'],'station_away','离岗超时',interval['start'],interval.get('review_frame') or interval.get('frame_start'),
                        f"{person['name']} 已观测离岗 {until-interval['start']:.1f} 秒；{interval.get('model_reason') or '已到设定阈值，等待大模型复核'}",pid,person.get('home_station'),until,status=status,threshold=self.policy['away_sec'])
            for status in self.station_intervals:
                sid=status['station_id'];check=status.get('ai_check');until=status['end'] if status['end'] is not None else self.time
                if not self._work_time(until if status['end'] is None else max(status['since'],until-1e-6)): continue
                if check and check.get('visible') is False:continue
                verified=bool(check and check['visible'] and check['count']==status['count'])
                disputed=bool(check and check['visible'] and check['count']!=status['count'])
                review_status='confirmed' if verified else 'rejected' if disputed else 'pending'
                if status['end'] is None and check and self.time-check['time_sec']>max(30,self.policy['review_interval_sec']*3):review_status='pending'
                duration=self._active_seconds(status['since'],until)
                station=next(s for s in self.scene['stations'] if s['id']==sid)
                if status['category']=='empty' and duration>=self.policy['empty_sec']:
                    add(f'{sid}-empty-{status["since"]}','station_empty','工位空缺',status['since'],(check or {}).get('frame_id') or status.get('frame_id'),f"{station['name']} 人数低于预期 {duration:.1f} 秒；{(check or {}).get('reason') or '已到设定阈值，待大模型复核'}",station=sid,end=until,interval_id=status['interval_id'],status=review_status,threshold=self.policy['empty_sec'])
                elif status['category']=='over' and duration>=self.policy['confirm_sec']:
                    add(f'{sid}-over-{status["since"]}','station_over_capacity','工位人数超限',status['since'],(check or {}).get('frame_id') or status.get('frame_id'),f"{station['name']} 观测 {status['count']} 人，预期 {station['expected_people']} 人；{(check or {}).get('reason') or '人数超限，待大模型复核'}",station=sid,end=until,interval_id=status['interval_id'],status=review_status,threshold=self.policy['confirm_sec'])
            return fresh

    def _active_seconds(self,start,end):
        """Exclude configured breaks from alert thresholds, preserving physical totals."""
        if end<=start:return 0.
        hours=self.policy['work_hours']+self.policy['break_hours']
        if not hours:return end-start
        if self.wall_origin is not None:
            clock=dt.datetime.fromtimestamp(self.wall_origin,dt.timezone(dt.timedelta(hours=8)))
            origin=clock.hour*3600+clock.minute*60+clock.second+clock.microsecond/1e6
        elif self.scene['start_clock']:
            h,m=map(int,self.scene['start_clock'].split(':'));origin=h*3600+m*60
        else:return 0.
        boundaries={start,end}
        for day in range(math.floor((origin+start)/86400),math.floor((origin+end)/86400)+1):
            for pair in hours:
                for clock in pair:
                    h,m=map(int,clock.split(':'));at=day*86400+h*3600+m*60-origin
                    if start<at<end:boundaries.add(at)
        points=sorted(boundaries)
        return sum(b-a for a,b in zip(points,points[1:]) if self._work_time((a+b)/2))

    def finish(self):
        with self.lock:
            if self.time is None:return
            for span in [*self.intervals,*self.station_intervals]:
                if span['end'] is None:span['end']=self.time

    def bind(self, pid, name, station_id=None):
        with self.lock:
            if pid not in self.people or not isinstance(name,str) or not name.strip() or len(name)>60: raise ValueError('人员或名称无效')
            if station_id is not None and station_id not in [s['id'] for s in self.scene['stations']]: raise ValueError('关联工位无效')
            self.people[pid]['name']=name.strip()
            self.people[pid]['identity_status']='manual';self.people[pid]['pending_links']=[]
            if station_id: self.people[pid]['home_station']=station_id

    def merge(self, source, target):
        with self.lock:
            if source==target or source not in self.people or target not in self.people: raise ValueError('人员关联无效')
            # Only link non-overlapping sightings; concurrent people cannot merge.
            a,b=self.people[source],self.people[target]
            if not (a['first_seen']>b.get('last_seen',self.time) or b['first_seen']>a.get('last_seen',self.time)):
                raise ValueError('两个人员观测时间重叠，不能合并')
            earlier,later=(target,source) if a['first_seen']>b['first_seen'] else (source,target)
            boundary=self.people[later]['first_seen']
            self.intervals[:]=[i for i in self.intervals if i['person_id']!=earlier or i['start']<boundary]
            for interval in self.intervals:
                if interval['person_id']==earlier and (interval['end'] is None or interval['end']>boundary):
                    interval['end']=boundary
                    if interval.get('frame_end_time',0)>boundary:
                        interval['frame_end']=interval.get('frame_start');interval['frame_end_time']=interval.get('frame_start_time')
            for interval in self.intervals:
                if interval['person_id'] in (source,target): interval['person_id']=target
            latest=copy.deepcopy(self.people[later]);latest['person_id']=target;latest['name']=b['name'];latest['identity_status']='manual';latest['pending_links']=[]
            latest['interval']=self.people[later]['interval']
            latest['home_station']=b.get('home_station') or a.get('home_station')
            latest['first_seen']=min(a['first_seen'],b['first_seen'])
            latest['linked_ids']=list(dict.fromkeys([source,target,*a.get('linked_ids',[]),*b.get('linked_ids',[])]))
            self.aliases[source]=target
            self.people[target]=latest;self.people.pop(source)
            return target

    def snapshot(self, include_intervals=False):
        with self.lock:
            now=self.time or 0; people=[];by_person={}
            for interval in self.intervals:by_person.setdefault(interval['person_id'],[]).append(interval)
            for pid,p in self.people.items():
                spans=by_person.get(pid,[])
                totals={state:sum(max(0,(i['end'] if i['end'] is not None else now)-i['start']) for i in spans if i['state']==state) for state in ('at_station','away','unknown')}
                away=[i for i in spans if i['state']=='away']
                episodes={i.get('away_episode') or i['interval_id'] for i in away}
                episode_durations={}
                for i in away:
                    key=i.get('away_episode') or i['interval_id']
                    episode_durations[key]=episode_durations.get(key,0)+max(0,(i['end'] if i['end'] is not None else now)-i['start'])
                current=p['interval']
                reasons={}
                for i in spans:
                    if i['state']=='unknown':
                        code=i.get('unknown_reason_code','legacy_unclassified')
                        reasons[code]=reasons.get(code,0)+max(0,(i['end'] if i['end'] is not None else now)-i['start'])
                people.append({k:copy.deepcopy(v) for k,v in p.items() if k not in ('interval','pending')} | {
                    'state':current['state'],'current_since':current['start'],'current_duration_sec':max(0,now-current['start']),
                    'at_station_sec':totals['at_station'],'away_sec':totals['away'],'unknown_sec':totals['unknown'],
                    'away_count':len(episodes),'longest_away_sec':max(episode_durations.values(),default=0),
                    'review_status':current.get('review_status','pending'),'model_reason':current.get('model_reason',''),
                    'unknown_reason':current.get('unknown_reason') or ('；'.join(UNKNOWN_REASONS.get(code,'旧版本未记录原因') for code,sec in reasons.items() if sec>0) if reasons else ''),
                    'unknown_breakdown':[{'code':code,'reason':UNKNOWN_REASONS.get(code,'旧版本未记录原因'),'duration_sec':sec} for code,sec in reasons.items() if sec>0],
                    'transition_pending':bool(p.get('pending'))})
            confirmed={s['id']:sum(p['state']=='at_station' and p.get('home_station')==s['id'] and p.get('visible',False) for p in people) for s in self.scene['stations']}
            result={'time_sec':now,'people':people,'view_observable':getattr(self,'view_observable',True),'stations':[{**s,'observed_count':self.counts[s['id']] if getattr(self,'view_observable',True) else None,
                      'resident_count':confirmed[s['id']],'ai_check':self.station_checks.get(s['id'])} for s in self.scene['stations']],
                    'observed_people':sum(self.counts.values()) if getattr(self,'view_observable',True) else None,'resident_people':sum(confirmed.values()),
                    'max_observed_people':self.max_observed_people,
                    'alerts':copy.deepcopy([a for a in self.alerts if a.get('review_status')!='rejected'][-500:]),'alarm_history':copy.deepcopy(self.alerts[-500:]),'gaps':copy.deepcopy(self.gaps[-100:]),'aliases':dict(self.aliases),'scheduled_active':self._work_time() if self.time is not None else False,
                    'timeline':copy.deepcopy(self.intervals[-200:]),'station_timeline':copy.deepcopy(self.station_intervals[-200:]),'policy':self.policy,
                    'note':'时间来自连续抽样观测；未知区间单列。在岗指位于关联工位区域，匿名编号不代表实名身份。'}
            if include_intervals:
                result['intervals']=copy.deepcopy(self.intervals);result['station_intervals']=copy.deepcopy(self.station_intervals)
            return result


class StationStore:
    def __init__(self, root: Path):
        self.root=root.resolve();self.lock=threading.RLock()

    def _db(self):
        self.root.mkdir(parents=True,exist_ok=True)
        db=sqlite3.connect(self.root/'sessions.sqlite3',timeout=10)
        db.execute('CREATE TABLE IF NOT EXISTS sessions (job_id TEXT PRIMARY KEY, revision TEXT, config TEXT, summary TEXT)')
        return db

    def save(self, job_id, revision, config, summary):
        with self.lock,closing(self._db()) as db,db:
            db.execute('INSERT OR REPLACE INTO sessions VALUES (?,?,?,?)',(job_id,revision,json.dumps(config,ensure_ascii=False,allow_nan=False),json.dumps(summary,ensure_ascii=False,allow_nan=False)))

    def load(self, job_id, revision=''):
        with self.lock,closing(self._db()) as db,db:
            row=db.execute('SELECT revision, config, summary FROM sessions WHERE job_id=?',(job_id,)).fetchone()
        if not row: raise FileNotFoundError('工位统计不存在')
        if revision and row[0]!=revision: raise ValueError('工位分析版本已更新')
        return {'job_id':job_id,'analysis_revision':row[0],'scene':json.loads(row[1]),**json.loads(row[2])}

    def delete(self, job_id):
        with self.lock,closing(self._db()) as db,db: db.execute('DELETE FROM sessions WHERE job_id=?',(job_id,))

    @contextmanager
    def deleting(self, job_id):
        """Acquire the database write before removing media; roll back on failure."""
        with self.lock,closing(self._db()) as db,db:
            db.execute('DELETE FROM sessions WHERE job_id=?',(job_id,))
            yield
