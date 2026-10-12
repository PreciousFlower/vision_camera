"""Ground visual QA in the workstation ledger's numbers and registered frames."""
import json
import re


def prepare_question(report, question):
    people=report.get('people') or []
    requested={p.upper() for p in re.findall(r'\bP\d{3,12}\b',question,re.I)}
    known={p['person_id'] for p in people}
    if requested-known: raise ValueError('未找到指定人员编号，请使用统计表中的编号')
    named=[p for p in people if p.get('name') and p['name']!=p['person_id'] and p['name'] in question]
    selected=[p for p in people if p['person_id'] in requested] if requested else named or people[:12]
    ids={p['person_id'] for p in selected}
    spans=[i for i in report.get('intervals',[]) if i['person_id'] in ids]
    regions=report.get('stations') or []
    mentioned=[s for s in regions if s['name'] in question or s['id'] in question]
    region_spans=[s for s in report.get('station_intervals',[]) if not mentioned or s['station_id'] in {v['id'] for v in mentioned}]
    ranked=sorted(spans,key=lambda i:-( (i['end'] if i['end'] is not None else report['time_sec'])-i['start'])) if '最长' in question else list(reversed(spans))
    if any(word in question for word in ('离岗','离开','返回')): ranked=[i for i in ranked if i['state']=='away'] or ranked
    refs=[]
    for i in ranked[:6]:refs.extend([i.get('frame_start'),i.get('frame_end')])
    for s in region_spans[-4:]:refs.extend([s.get('frame_id'),s.get('frame_end')])
    refs=[fid for fid in dict.fromkeys(refs) if isinstance(fid,str) and re.fullmatch(r'f_\d+',fid)][:18]
    facts={'source':'程序按采集/视频时间连续抽样累计，未知区间单列；不包含视频开始前的时间',
           'time_sec':report['time_sec'],'observed_people':report.get('observed_people'),
           'max_observed_people':report.get('max_observed_people'),'use_mock':report.get('use_mock',False),
           'people':[{k:p.get(k) for k in ('person_id','name','home_station','state','current_duration_sec','at_station_sec','away_sec','unknown_sec','away_count','longest_away_sec')} for p in selected],
           'stations':[{k:s.get(k) for k in ('id','name','observed_count','resident_count','expected_people')} for s in (mentioned or regions)[:20]],
           'person_intervals':[{k:i.get(k) for k in ('person_id','state','station_id','start','end','review_status')} for i in ranked[:30]],
           'station_intervals':[{k:s.get(k) for k in ('station_id','category','count','since','end')} for s in region_spans[-30:]],
           'limitations':'单摄像头匿名关联；短时遮挡、漏采及身份不确定不代表真实离岗，人数为可观测结果。'}
    context=json.dumps({'trusted_workstation_statistics':facts},ensure_ascii=False,allow_nan=False)
    if len(context)>24000:raise ValueError('工位统计过多，请指定人员或工位提问')
    return facts,refs,context


def validate_numbers(answer, facts, evidence=None):
    seconds={float(facts['time_sec'])};counts=set()
    seconds.update(float(e['time_sec']) for e in evidence or [] if isinstance(e.get('time_sec'),(int,float)))
    for p in facts['people']:
        seconds.update(float(p[k]) for k in ('current_duration_sec','at_station_sec','away_sec','unknown_sec','longest_away_sec') if p.get(k) is not None)
        counts.add(p['away_count'])
    for i in facts['person_intervals']:
        seconds.add(float(i['start']))
        end=facts['time_sec'] if i['end'] is None else i['end'];seconds.update((float(end),float(end)-float(i['start'])))
    for s in facts['station_intervals']:
        end=facts['time_sec'] if s['end'] is None else s['end'];seconds.update((float(s['since']),float(end),float(end)-float(s['since'])))
    for s in facts['stations']:counts.update(s[k] for k in ('observed_count','resident_count','expected_people') if s.get(k) is not None)
    counts.update(v for v in (facts.get('observed_people'),facts.get('max_observed_people')) if v is not None)
    text=answer['answer']
    for number in re.findall(r'(\d+(?:\.\d+)?)\s*(?:秒|s\b)',text):
        value=float(number)
        if not any(abs(value-round(v,1))<=.11 for v in seconds):raise ValueError('模型回答中的秒数与已保存工位统计不符，请指定人员后重试')
    if re.search(r'\d+(?:\.\d+)?\s*(?:分钟|小时)',text):raise ValueError('模型未按统计秒数返回时长，请重试')
    for count in re.findall(r'(\d+)\s*(?:次|人)',text):
        if int(count) not in counts:raise ValueError('模型回答中的次数或人数与工位统计不符')
    known={p['person_id'] for p in facts['people']}
    if {p.upper() for p in re.findall(r'\bP\d{3,12}\b',text,re.I)}-known:raise ValueError('模型引用了本次统计未包含的人员')
    return answer
