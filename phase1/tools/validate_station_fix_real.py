"""Opt-in visual checks on user-cached workstation evidence; owns one bounded GPU.

Run explicitly with --run. Sparse saved frames validate occupancy and reference
handling, not continuous duration accuracy or biometric identity recognition.
Raw results stay in ignored runs/; account jobs are checked before allocation and
the single owned service is released in finally, including failed model checks.
"""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
import re
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))


def cached_cases(job_id, replay_path, output):
    import cv2
    from webui.workstation_runner import annotate, spatial_evidence

    folder = ROOT / 'runs/video_analyses' / job_id
    metadata = json.loads((folder / 'metadata.json').read_text(encoding='utf-8'))
    registered = {}
    for line in (folder / 'frame_catalog.jsonl').read_text(encoding='utf-8').splitlines():
        row = json.loads(line)
        index, at = row['index'], row['time_sec']
        if index in registered and abs(registered[index] - at) > 1e-6:
            raise ValueError('Registered evidence has conflicting capture times')
        registered[index] = at
    replay = json.loads(replay_path.read_text(encoding='utf-8'))
    samples = replay.get('samples', replay.get('read_only_keyframe_replay', replay.get('frames', []))) if isinstance(replay, dict) else replay
    by_frame = {s['frame_id']: s for s in samples}
    stations = metadata['station_scene']['stations']
    # These are actual registered frames from the user's reported experiment.
    # Expectations are visual occupancy only; no employee identity assertion.
    cases = []
    for label, frame_id, expected in (('occupied', 'f_0', 1), ('empty', 'f_244', 0), ('returned', 'f_559', 1)):
        sample = by_frame.get(frame_id)
        if sample is None:
            raise ValueError(f'Replay lacks required real evidence {frame_id}')
        index = int(frame_id[2:])
        if index not in registered or abs(sample['time_sec'] - registered[index]) > 1e-6:
            raise ValueError('Replay frame was not registered with this capture time')
        image = cv2.imread(str(ROOT / 'data/frames' / job_id / f'f_{index:05d}.jpg'))
        if image is None:
            raise ValueError('Saved original evidence JPEG is unavailable')
        tracks = sample['tracks']
        # The anonymous candidate must come from the real detector replay.
        # Never rename an arbitrary detection to P001 just to pass this test.
        if expected and {t['person_id'] for t in tracks} != {'P001'}:
            raise ValueError('Real replay did not uniquely associate the returning candidate')
        if not expected and tracks:
            raise ValueError('Selected empty image contains detector candidates; inspect before testing')
        marked = annotate(image, stations, tracks)
        ok, encoded = cv2.imencode('.jpg', marked)
        if not ok:
            raise ValueError('Review image JPEG encoding failed')
        (output / f'{label}-{frame_id}.jpg').write_bytes(encoded.tobytes())
        evidence = spatial_evidence({'frame_id': frame_id, 'timestamp': registered[index],
                                     'index': index, 'tracks': tracks}, stations, {'P001'})
        cases.append({'label': label, 'expected_count': expected, 'frames': [base64.b64encode(encoded).decode('ascii')],
                      'evidence': [evidence], 'tracks': tracks})
    people = [{'person_id': 'P001', 'home_station': stations[0]['id'], 'identity_status': 'matched',
               'name': '匿名人员（本地衣着候选关联，并非实名身份识别）'}]
    return metadata, stations, people, cases


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true', help='Explicitly allocate one remote GPU for these visual checks')
    parser.add_argument('--job-id', default='5d4baefe1e9e', help='Local analyzed-video identifier')
    parser.add_argument('--replay', type=Path, default=ROOT / 'runs/_workstation_fix_20261013/tracking-replay.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'runs/_workstation_fix_20261013/real-model-validation')
    args = parser.parse_args(argv)
    if not args.run:
        print('Not run: pass --run to explicitly create one bounded validation allocation.', flush=True)
        return 0
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', args.job_id):
        parser.error('Local job identifier is invalid')

    from check_validation_cluster import configure_auth
    from civility.config import load_settings, load_behaviors, behavior_payload
    from webui.cluster_client import ClusterClient, ServiceSession, deploy_service_package

    configure_auth()
    cc = load_settings()['cluster']
    client = ClusterClient(cc['remote_workdir'], cc['remote_python'], cc['remote_model_root'])
    # Read account state before doing work or writing a service package.
    existing = client.list_my_jobs()
    if existing:
        print('Skipped: account already has allocations:', json.dumps(existing), flush=True)
        return 2
    # Keep prior failed-counterexample and passing raw traces reviewable even
    # when the same opt-in command is used again with its default output path.
    if (args.output / 'result.json').exists():
        args.output = args.output / ('run-' + uuid.uuid4().hex[:8])
    args.output.mkdir(parents=True, exist_ok=True)
    metadata, stations, people, cases = cached_cases(args.job_id, args.replay, args.output)
    session = ServiceSession(client, cc['remote_model'], port=18194, exclude=cc['exclude_nodes'],
                             extra_sbatch='--time=00:12:00')
    result = {'status': 'prepared', 'source_job_id': args.job_id, 'analysis_revision': metadata['analysis_revision'],
              'validation_job_id': None, 'cases': [],
              'limitations': 'Sparse saved frames validate 1→0→1 ROI occupancy and real frame references. '
                             'They do not validate continuous durations or prove real employee identity. '
                             'Old failed-window raw output was not archived, so its exact malformed content cannot be replayed.'}

    def persist(**extra):
        result.update(extra)
        if session.jobid:
            result['validation_job_id'] = session.jobid
        (args.output / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')

    failure = None
    try:
        deploy_service_package(client, ROOT, behavior_payload(load_behaviors()))
        if client.list_my_jobs():
            raise RuntimeError('Account changed during preparation; no duplicate allocation was created')
        session.start(wait_ready_sec=240, progress=lambda message: persist(status=message))
        health = session.health()
        persist(status='checking', health={k: health.get(k) for k in ('model', 'device', 'workstation_protocol')})
        if health.get('workstation_protocol') != 1:
            raise RuntimeError('New workstation review protocol is not loaded')
        mismatches=[]
        for case in cases:
            review = session.review_workstations(case['frames'], case['evidence'], stations, people, metadata['profile'])
            case_result = {'label': case['label'], 'expected_count': case['expected_count'],
                           'evidence': case['evidence'], 'tracks': case['tracks'], 'review': review}
            result['cases'].append(case_result)
            persist()
            checks = review['parsed']['stations']
            matching = [s for s in checks if s['station_id'] == stations[0]['id']
                        and s['frame_id'] == case['evidence'][0]['frame_id']]
            if not any(s['visible'] and s['count'] == case['expected_count'] for s in matching):
                mismatches.append(case['label'])
            print(json.dumps({'label': case['label'], 'frame_ids': [e['frame_id'] for e in case['evidence']],
                              'counts': [{'frame_id': s['frame_id'], 'visible': s['visible'], 'count': s['count']} for s in matching],
                              'tokens_out': review.get('tokens_out'), 'warnings': review.get('protocol_warnings', [])}), flush=True)
        if mismatches:
            raise RuntimeError('Visual occupancy mismatch in '+', '.join(mismatches)+'; actual results remain archived')
        persist(status='passed')
    except Exception as exc:
        failure = exc
        persist(status='failed', error=f'{type(exc).__name__}: {exc}',
                failed_diagnostic=getattr(exc, 'station_diagnostic', {}))
    finally:
        owned = session.jobid
        try:
            released = session.stop(sweep_orphans=False)
            remaining = client.list_my_jobs()
            deadline = time.monotonic() + 60
            while any(j['jobid'] == owned for j in remaining) and time.monotonic() < deadline:
                time.sleep(2)
                remaining = client.list_my_jobs()
            persist(released_job_ids=released, remaining_allocations=remaining)
            print('Validation allocation released; remaining:', json.dumps(remaining), flush=True)
            if any(j['jobid'] == owned for j in remaining):
                raise RuntimeError('Validation allocation still listed after release; inspect account state')
        except Exception as exc:
            persist(cleanup_error=f'{type(exc).__name__}: {exc}')
            raise
    if failure:
        raise failure
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
