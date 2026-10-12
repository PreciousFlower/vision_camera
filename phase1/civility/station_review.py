"""Visual workstation review protocol shared by local and deployed VLM callers."""
import json
import math


def validate_request(body):
    from .video_qa import validate_request as validate_qa
    request=validate_qa({**body,'question':'核对工位人员与画面证据'})
    if len(body['frames'])>8: raise ValueError('工位核验最多8张图片')
    for frame in request['evidence']:
        timestamp=frame.get('time_sec')
        if type(timestamp) not in (int,float) or not math.isfinite(timestamp) or timestamp<0: raise ValueError('证据缺少可靠采集时间')
    stations,people=body.get('stations'),body.get('people')
    if not isinstance(stations,list) or not 1<=len(stations)<=6 or not isinstance(people,list) or len(people)>12:
        raise ValueError('单次工位核验最多6个工位和12个人员')
    station_ids=[];person_ids=[]
    for s in stations:
        if not isinstance(s,dict) or not isinstance(s.get('id'),str) or not s['id'] or len(s['id'])>32: raise ValueError('工位编号无效')
        polygon=s.get('polygon')
        if not isinstance(polygon,list) or not 3<=len(polygon)<=20 or any(not isinstance(p,list) or len(p)!=2 or any(type(v) not in (int,float) or not math.isfinite(v) or not 0<=v<=1 for v in p) for p in polygon): raise ValueError('工位坐标无效')
        station_ids.append(s['id'])
    for p in people:
        if not isinstance(p,dict) or not isinstance(p.get('person_id'),str) or len(p['person_id'])>32 or not p['person_id']: raise ValueError('人员编号无效')
        person_ids.append(p['person_id'])
    if len(set(station_ids))!=len(station_ids) or len(set(person_ids))!=len(person_ids): raise ValueError('工位或人员编号重复')
    for frame in request['evidence']:
        hints=frame.get('geometry_hint',[])
        if not isinstance(hints,list) or len(hints)>12:raise ValueError('人员定位提示过多')
        for hint in hints:
            if not isinstance(hint,dict) or hint.get('person_id') not in person_ids:raise ValueError('定位提示人员无效')
            for key,size in (('bbox',4),('body_center',2)):
                values=hint.get(key)
                if not isinstance(values,list) or len(values)!=size or any(type(v) not in (int,float) or not math.isfinite(v) or not 0<=v<=1 for v in values):raise ValueError('定位提示坐标无效')
            candidates=hint.get('candidate_stations')
            if not isinstance(candidates,list) or any(sid not in station_ids for sid in candidates):raise ValueError('定位提示工位无效')
    rules=body.get('behavior_rules',[])
    if not isinstance(rules,list) or len(rules)>3 or any(not isinstance(r,dict) or not isinstance(r.get('definition'),str) or len(r['definition'])>5000 for r in rules):raise ValueError('工位行为说明无效')
    return {**request,'task_mode':'workstation','stations':stations,'people':people,'behavior_rules':rules}


def build_messages(images, request):
    system=(
        '你是视频区域人员复核助手。唯一视觉任务是核对用户指定蓝色多边形区域内有几名真实人员，并确认匿名编号对应的人是否可见。'
        '“工位”仅是区域的名字，不要求画面是办公室，不要求人在工作。站在街边、拿物品、抽烟等都不改变区域内外的几何位置。'
        '蓝色工位边界是用户配置，不是事实证据；框和人员编号用于定位，检测框可能漏人或出错。'
        '必须独立看图，不能仅相信检测器的位置说明。短暂路过可标passing，遮挡标occluded，身份不能确认标uncertain。'
        '工位就是用户指定的多边形统计区域，坐标x/y均为相对整张图片的0至1比例。'
        '人数按人物身体中心是否在多边形内部计数，包含区域内的路过人员；驻留状态另行判断。'
        '每张图的geometry_hint是检测器提供的候选框、身体中心和候选区域，帮助定位，不是已经确认的人。'
        '先看框内是否确实有人、对应编号是否可辨认，再依据位置核对区域；未见真实人员时不要照抄提示。'
        '即使区域里没有办公桌、椅子，也不能擅自把已配置的区域人数判为零。'
        'at_station指人员驻留在指定区域，并非认定其工作效率。outside需看见对应人员确实离开指定工位；'
        '单纯消失不能当作离岗。人数包括清楚可见的实际人员；无法看清整个工位则visible=false，count=null。'
        '匿名人员编号不是姓名。不得猜测员工身份。时间和frame_id只引用输入图片标签，不计算累计时长。'
        '报告最后一张图状态，同时检查前后过程；如发生离开或返回，可为同一人员额外报告一个关键历史帧状态，每人最多两条不同帧观察。工位人数发生变化时，也可补充该工位一个历史帧人数，每个工位最多两条。'
        '只输出JSON：{"scene":"简短说明","observations":[{"person_id":"输入编号",'
        '"station_id":"输入工位编号或null","state":"at_station|outside|occluded|passing|uncertain",'
        '"frame_id":"实际图片编号","reason":"可见依据"}],"stations":[{"station_id":"输入编号",'
        '"visible":true,"count":1,"frame_id":"实际图片编号","reason":"人数及区域可见依据"}]}。'
        '只使用输入中的人员和工位编号，理由各不超过40字。未提供足够画面时明确不确定，不能补写进出动作。'
    )
    stations=[]
    for station in request['stations']:
        polygon=station['polygon'];xs=[p[0] for p in polygon];ys=[p[1] for p in polygon]
        full=len(polygon)==4 and set(map(tuple,polygon))=={(0,0),(1,0),(1,1),(0,1)}
        stations.append({**station,'scope_description':'整张图片都是该区域，图片中的人员均处于该区域内' if full else f'图中蓝色多边形，外接范围x={min(xs):.3f}至{max(xs):.3f}、y={min(ys):.3f}至{max(ys):.3f}；需以多边形为准'})
    content=[{'type':'text','text':json.dumps({'stations':stations,'people':request['people'],'behavior_rules':request.get('behavior_rules',[])},ensure_ascii=False)}]
    for image,evidence in zip(images,request['evidence']):
        content.extend([{'type':'text','text':json.dumps(evidence,ensure_ascii=False)},{'type':'image','image':image}])
    return [{'role':'system','content':system},{'role':'user','content':content}]


def validate_result(obj, request):
    if not isinstance(obj,dict) or not isinstance(obj.get('observations'),list) or not isinstance(obj.get('stations'),list): raise ValueError('工位核验缺少结构化结果')
    if len(obj['observations'])>24 or len(obj['stations'])>12: raise ValueError('工位核验返回过多条目')
    frames={e['frame_id'] for e in request['evidence']};people={p['person_id'] for p in request['people']};stations={s['id'] for s in request['stations']}
    output={'scene':str(obj.get('scene') or '')[:1000],'observations':[],'stations':[]}
    seen=set()
    for item in obj['observations']:
        if not isinstance(item,dict) or not isinstance(item.get('person_id'),str) or item.get('person_id') not in people: raise ValueError('模型人员引用无效')
        if not isinstance(item.get('frame_id'),str) or not isinstance(item.get('state'),str): raise ValueError('人员状态或画面引用无效')
        if item.get('frame_id') not in frames or item.get('state') not in ('at_station','outside','occluded','passing','uncertain'): raise ValueError('人员状态或画面引用无效')
        key=(item['person_id'],item['frame_id'])
        sid=item.get('station_id')
        if sid is not None and (not isinstance(sid,str) or sid not in stations): raise ValueError('模型工位引用无效')
        if item['state']=='at_station' and sid is None: raise ValueError('在岗状态必须引用工位')
        if not isinstance(item.get('reason'),str) or not item['reason'].strip(): raise ValueError('人员复核缺少证据理由')
        normalized={k:item[k] for k in ('person_id','state','frame_id','reason')}|{'station_id':sid}
        if key in seen:
            previous=next(o for o in output['observations'] if (o['person_id'],o['frame_id'])==key)
            if (previous['state'],previous['station_id'])!=(normalized['state'],sid):
                previous.update(state='uncertain',station_id=None,reason='模型对同一人员和画面的判断冲突，需复核')
        else:seen.add(key);output['observations'].append(normalized)
    seen=set()
    for item in obj['stations']:
        if not isinstance(item,dict) or not isinstance(item.get('station_id'),str) or item.get('station_id') not in stations: raise ValueError('工位引用无效')
        if not isinstance(item.get('frame_id'),str): raise ValueError('工位画面引用无效')
        if item.get('frame_id') not in frames or type(item.get('visible')) is not bool: raise ValueError('工位可见性或画面引用无效')
        key=(item['station_id'],item['frame_id'])
        count=item.get('count')
        if (item['visible'] and (type(count) is not int or not 0<=count<=30)) or (not item['visible'] and count is not None): raise ValueError('工位人数无效')
        if not isinstance(item.get('reason'),str) or not item['reason'].strip(): raise ValueError('工位复核缺少证据理由')
        normalized={k:item[k] for k in ('station_id','visible','count','frame_id','reason')}
        if key in seen:
            previous=next(s for s in output['stations'] if (s['station_id'],s['frame_id'])==key)
            if (previous['visible'],previous['count'])!=(normalized['visible'],count):
                previous.update(visible=False,count=None,reason='模型对同一工位和画面的人数判断冲突，需复核')
        else:seen.add(key);output['stations'].append(normalized)
    if not output['stations']: raise ValueError('模型未复核任何工位')
    return output
