"""Versioned behavior presets. Tasks use immutable snapshots, editors use drafts."""
from __future__ import annotations
import copy
import hashlib
import json
import os
import re
import threading
import uuid
from pathlib import Path

from civility.config import behavior_payload, load_behaviors, parse_behaviors

PRESETS = (
    {'id': 'current', 'name': '当前综合识别', 'mode': 'behavior'},
    {'id': 'construction', 'name': '施工安全', 'mode': 'behavior'},
    {'id': 'workstations', 'name': '工位人员管理', 'mode': 'workstation'},
)
STATION_DEFAULTS = {'sample_fps': 2, 'confirm_sec': 3, 'max_gap_sec': 2.5,
                    'occlusion_sec': 3, 'review_interval_sec': 10, 'away_sec': 60,
                    'empty_sec': 60, 'work_hours': [], 'break_hours': []}


class ProfileRegistry:
    def __init__(self, root: Path, current_path: Path):
        self.root, self.current_path = root.resolve(), current_path
        self.lock = threading.RLock()

    def path(self, profile_id):
        if not isinstance(profile_id, str) or not re.fullmatch(r'[a-z][a-z0-9_-]{0,47}', profile_id):
            raise ValueError('预设编号无效')
        return self.root / (profile_id + '.json')

    def load(self, profile_id):
        path = self.path(profile_id)
        with self.lock:
            if path.exists():
                doc = json.loads(path.read_text(encoding='utf-8'))
            else:
                preset = next((p for p in PRESETS if p['id'] == profile_id), None)
                if not preset:
                    raise FileNotFoundError('行为预设不存在')
                doc = copy.deepcopy(preset)
                bs = load_behaviors(self.current_path)
                if profile_id == 'construction':
                    bs.behaviors = [b for b in bs.behaviors if b.id in ('no_safety_helmet', 'no_helmet')]
                if profile_id == 'workstations':
                    doc['behaviors'] = [{'id': bid, 'name': name, 'enabled': True, 'severity': 'medium',
                                        'definition': definition, 'evidence_required': '需核对划定工位和对应时刻的实际画面；遮挡、断流及身份不确定不能当作明确离岗。',
                                        'counter_examples': ['正常工作时段之外', '短暂路过或遮挡', '人员身份尚未关联']}
                        for bid, name, definition in (
                            ('station_away', '离岗超时', '已关联工位的同一人员明确离开指定工位，已观测离岗时长达到配置阈值。'),
                            ('station_empty', '工位空缺', '可清楚观察的工位持续人数低于配置预期人数，达到空缺阈值。'),
                            ('station_over_capacity', '工位人数超限', '划定工位内人数持续高于配置预期人数。'))]
                    doc['defaults'] = {'max_new_tokens': 1536, 'few_shot': False}
                    doc['station_policy'] = copy.deepcopy(STATION_DEFAULTS)
                else:
                    doc.update(behavior_payload(bs))
            return self.validate(doc)

    def validate(self, doc):
        if not isinstance(doc, dict): raise ValueError('预设必须是对象')
        self.path(doc.get('id'))
        name = doc.get('name')
        if not isinstance(name, str) or not name.strip() or len(name) > 60: raise ValueError('预设名称须为1至60字')
        if doc.get('mode') not in ('behavior', 'workstation'): raise ValueError('预设模式无效')
        if not isinstance(doc.get('defaults',{}),dict) or not isinstance(doc.get('behaviors'),list) or len(doc['behaviors'])>40: raise ValueError('预设默认配置或行为列表无效')
        if any(not isinstance(b,dict) or ('enabled' in b and type(b['enabled']) is not bool) for b in doc['behaviors']):raise ValueError('行为启用项必须是布尔值')
        bs = parse_behaviors(doc)
        if not bs.behaviors: raise ValueError('预设至少包含一项行为')
        result = {'id': doc['id'], 'name': name.strip(), 'mode': doc['mode'], **behavior_payload(bs)}
        # Preserve disabled definitions in the editor and archived snapshots.
        from dataclasses import asdict
        result['behaviors'] = [asdict(b) for b in bs.behaviors]
        if doc['mode'] == 'workstation':
            if any(b.id not in ('station_away','station_empty','station_over_capacity') for b in bs.behaviors):raise ValueError('工位预设支持离岗超时、工位空缺、人数超限；普通行为请使用综合或施工预设')
            tokens=doc.get('defaults',{}).get('max_new_tokens',1536)
            if type(tokens) is not int or not 256<=tokens<=1536:raise ValueError('工位输出预算须为256至1536')
            from .workstations import validate_policy
            result['station_policy'] = validate_policy(doc.get('station_policy') or STATION_DEFAULTS)
        result['version'] = hashlib.sha256(json.dumps(result, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:16]
        return result

    def save(self, profile_id, doc):
        with self.lock:
            result = self.validate({**doc, 'id': profile_id})
            if profile_id == 'current' and result['mode'] != 'behavior': raise ValueError('当前综合识别的模式不能改变')
            self.root.mkdir(parents=True, exist_ok=True)
            path = self.path(profile_id)
            temp = path.with_suffix('.' + uuid.uuid4().hex + '.tmp')
            try:
                temp.write_text(json.dumps(result, ensure_ascii=False, allow_nan=False), encoding='utf-8')
                os.replace(temp, path)
            finally: temp.unlink(missing_ok=True)
            return result

    def list(self):
        with self.lock:
            ids = [p['id'] for p in PRESETS] + [p.stem for p in self.root.glob('*.json') if p.stem != '_default']
            profiles = []
            for profile_id in dict.fromkeys(ids):
                try:
                    doc = self.load(profile_id)
                    profiles.append({k: doc[k] for k in ('id', 'name', 'mode', 'version')})
                except (OSError, ValueError, TypeError): continue
            try: default = json.loads((self.root / '_default.json').read_text(encoding='utf-8'))['id']
            except (OSError, ValueError, KeyError): default = 'current'
            if default not in [p['id'] for p in profiles]: default = 'current'
            return {'profiles': profiles, 'default_id': default}

    def set_default(self, profile_id):
        with self.lock:
            self.load(profile_id)
            self.root.mkdir(parents=True, exist_ok=True)
            temp = self.root / ('_default-' + uuid.uuid4().hex + '.tmp')
            try:
                temp.write_text(json.dumps({'id': profile_id}), encoding='utf-8')
                os.replace(temp, self.root / '_default.json')
            finally: temp.unlink(missing_ok=True)

    def delete(self, profile_id):
        with self.lock:
            if profile_id in [p['id'] for p in PRESETS]: raise ValueError('内置预设请复制后修改，不能删除')
            selected=self.list()['default_id']==profile_id
            self.path(profile_id).unlink()
            if selected: self.set_default('current')
