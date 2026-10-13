"""Keep manual anonymous identity links consistent at model request boundaries.

The tracker owns its local track IDs. The ledger owns explicit human links.
Buffered frames and in-flight model responses are snapshots from before those
links, so both sides of a request need the same canonicalization. Original
images/model text stay available separately as evidence; this module never
infers that two unrelated people are the same person.
"""
from __future__ import annotations

import copy
import json
import re


_PERSON_REFERENCE = re.compile(r'(?<![A-Za-z0-9_])P\d{3,}(?![A-Za-z0-9_])')


def canonical_id(person_id, aliases):
    """Resolve only explicit ledger links; reject corrupt cycles."""
    visited = set()
    while person_id in aliases:
        if person_id in visited:
            raise ValueError('人工人员关联存在循环，请重新关联')
        visited.add(person_id)
        person_id = aliases[person_id]
    return person_id


def canonical_samples(samples, aliases):
    """Copy buffered geometry with IDs current at the time of submission."""
    output = copy.deepcopy(samples)
    for sample in output:
        for track in sample.get('tracks', []):
            track['person_id'] = canonical_id(track['person_id'], aliases)
            pending = [canonical_id(pid, aliases) for pid in track.get('pending_links', [])]
            track['pending_links'] = list(dict.fromkeys(pid for pid in pending if pid != track['person_id']))
            if not track['pending_links'] and track['person_id'] in aliases.values():
                track['identity_status'] = 'manual'
    return output


def canonical_review(parsed, aliases, allowed_person_ids):
    """Normalize display and semantic evidence after a possibly delayed reply.

Free-form text can still invent an ID even when the structured ID passed the
protocol validator. That cannot confirm an identity or a count. Retain the raw
reply outside this result and expose a diagnostic warning instead.
"""
    output = copy.deepcopy(parsed)
    allowed = {canonical_id(pid, aliases) for pid in allowed_person_ids}
    warnings = list(output.get('validation_warnings', []))

    def normalize_text(value, label):
        unknown = set()
        def replace(match):
            person_id = canonical_id(match.group(), aliases)
            if person_id not in allowed:
                unknown.add(match.group())
                return '未关联人员'
            return person_id
        text = _PERSON_REFERENCE.sub(replace, str(value or ''))
        if unknown:
            warnings.append(f'{label}: 原始描述含未提供的人员编号 {", ".join(sorted(unknown))}，身份待复核')
        return text, bool(unknown)

    output['scene'], scene_unknown = normalize_text(output.get('scene'), 'scene')
    if scene_unknown:
        output['scene'] = '人员编号待复核：' + output['scene']
    observations = []
    seen = {}
    for index, item in enumerate(output.get('observations', [])):
        item['person_id'] = canonical_id(item['person_id'], aliases)
        item['reason'], unknown = normalize_text(item.get('reason'), f'observations[{index}]')
        if item['person_id'] not in allowed:
            warnings.append(f'observations[{index}]: 未提供的人员编号，不能确认人员身份')
            continue
        if unknown:
            item.update(state='uncertain', station_id=None)
        key = (item['person_id'], item.get('frame_id'))
        previous = seen.get(key)
        if previous is None:
            seen[key] = item
            observations.append(item)
        elif (previous.get('state'), previous.get('station_id')) != (item.get('state'), item.get('station_id')):
            previous.update(state='uncertain', station_id=None, reason='人工关联后的同帧模型判断冲突，身份待复核')
            warnings.append(f'observations[{index}]: 人工关联后的同帧状态冲突，不能确认')
    output['observations'] = observations
    for index, item in enumerate(output.get('stations', [])):
        item['reason'], unknown = normalize_text(item.get('reason'), f'stations[{index}]')
        if unknown:
            item.update(visible=False, count=None)
    if warnings:
        output['validation_warnings'] = list(dict.fromkeys(warnings))[:72]
    return output


def canonical_window(row, aliases, known_person_ids=None):
    """Present previously archived windows using later explicit human links.

    This is a view transformation, not a rewrite of the original archive. The
    raw model output and protocol diagnostics remain verbatim, so viewers can
    inspect why an old label was mapped. Unknown-ID validation is only possible
    when the caller supplies the task's authoritative person roster.
    """
    output = copy.deepcopy(row)
    mapped = {}

    def map_text(value):
        def replace(match):
            original = match.group()
            person_id = canonical_id(original, aliases)
            if person_id != original:
                mapped[original] = person_id
            return person_id
        return _PERSON_REFERENCE.sub(replace, str(value or ''))

    output['scene'] = map_text(output.get('scene'))
    for person in output.get('people') or []:
        for key in ('appearance', 'changed', 'person_id'):
            if key in person:
                person[key] = map_text(person[key])
    parsed = output.get('station_review')
    if isinstance(parsed, dict):
        # Collect an audit mapping before semantic normalization modifies IDs.
        map_text(json.dumps(parsed, ensure_ascii=False))
        if known_person_ids is not None:
            parsed['scene'] = output['scene']
            normalized = canonical_review(parsed, aliases, known_person_ids)
            output['station_review'] = normalized
            output['scene'] = normalized['scene']
            frame_times = {f'f_{index}':timestamp for index,timestamp in
                           zip(output.get('frame_indices') or [], output.get('frame_times') or [])}
            reliable_times = all(item.get('frame_id') in frame_times for item in normalized['observations'])
            observations = sorted(normalized['observations'], key=lambda item:frame_times[item['frame_id']]) if reliable_times else normalized['observations']
            latest = {item['person_id']:item for item in observations}
            people = []
            shown = set()
            for person in output.get('people') or []:
                observation = latest.get(person.get('appearance'))
                if observation is not None and observation['person_id'] not in shown:
                    if reliable_times:
                        person.update(changed=observation['reason'], action=observation['state'],
                                      where=observation.get('station_id') or '工位外')
                    else:
                        # Old archives may lack frame_times. Their existing
                        # display already chose a current observation; unknown
                        # array order must not switch it to a historic state.
                        matched = next((item for item in observations if item['person_id']==observation['person_id']
                                        and item['state']==person.get('action')), None)
                        if matched is not None:
                            person['changed'] = matched['reason']
                    people.append(person)
                    shown.add(observation['person_id'])
            output['people'] = people
        else:
            parsed['scene'] = map_text(parsed.get('scene'))
            for item in [*(parsed.get('observations') or []), *(parsed.get('stations') or [])]:
                if 'person_id' in item:
                    item['person_id'] = canonical_id(item['person_id'], aliases)
                if 'reason' in item:
                    item['reason'] = map_text(item['reason'])
    if mapped:
        output['identity_mapping'] = [{'from':source, 'to':target} for source,target in sorted(mapped.items())]
        output['identity_note'] = '人员编号已按人工关联统一展示；原始模型输出保留当时编号。'
    return output


def canonical_alarm(alert, aliases):
    """Display existing alarm text through explicit links, retaining its audit.

    A text that mentions two former IDs now mapped to one may describe different
    times, or may contain a conflict. Do not rewrite it into a new claim that
    one person was both present and absent. Preserve that original text with a
    mapping note; never change confirmation/retraction status or status history.
    """
    output = copy.deepcopy(alert)
    mapped = {}
    collisions = []
    original_text = dict(output.get('original_identity_text') or {})

    def normalize(value, field):
        if not isinstance(value, str):
            return value
        references = set(_PERSON_REFERENCE.findall(value))
        by_target = {}
        for reference in references:
            target = canonical_id(reference, aliases)
            by_target.setdefault(target, set()).add(reference)
            if target != reference:
                mapped[reference] = target
        if any(len(originals)>1 for originals in by_target.values()):
            collisions.append(field)
            original_text.setdefault(field, value)
            return value
        normalized = _PERSON_REFERENCE.sub(lambda match:canonical_id(match.group(), aliases), value)
        if normalized != value:
            original_text.setdefault(field, value)
        return normalized

    person_id = output.get('person_id')
    if isinstance(person_id, str):
        target = canonical_id(person_id, aliases)
        if target != person_id:
            mapped[person_id] = target
            output.setdefault('original_person_id', person_id)
            output['person_id'] = target
    for field in ('reason', 'person_desc', 'latest_review_reason', 'retraction_reason', 'model_reason'):
        if field in output:
            original = output[field]
            output[field] = normalize(original, field)
            if field == 'reason' and field in original_text:
                output.setdefault('original_reason', original_text[field])
    if isinstance(output.get('evidences'), list):
        output['evidences'] = [normalize(value, f'evidences[{index}]') for index,value in enumerate(output['evidences'])]
    if mapped:
        existing = {entry['from']:entry['to'] for entry in output.get('identity_mapping') or []
                    if isinstance(entry, dict) and isinstance(entry.get('from'), str) and isinstance(entry.get('to'), str)}
        existing.update(mapped)
        output['identity_mapping'] = [{'from':source, 'to':target} for source,target in sorted(existing.items())]
        output['identity_note'] = ('人员编号按人工关联统一展示，未重新生成模型判断；'
                                   '原描述涉及多个关联前编号，已保留当时原文。' if collisions else
                                   '人员编号按人工关联统一展示；原始理由和当时的复核历史已保留，未重新生成模型判断。')
        if original_text:
            output['original_identity_text'] = original_text
        if collisions:
            output['identity_collision_fields'] = collisions
    return output
