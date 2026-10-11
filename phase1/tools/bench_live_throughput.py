"""Measure the real deployed model without opening a camera or changing window settings."""
from __future__ import annotations

import argparse
import base64
import concurrent.futures
import json
from pathlib import Path
import statistics
import sys
import time

import cv2
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from civility.config import behavior_payload, load_behaviors, load_settings
from civility.recognition import generation_budget, infer_with_review


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--url', help='Model endpoint; defaults to the currently running UI service')
    ap.add_argument('--ui', default='http://127.0.0.1:8848')
    ap.add_argument('--video', default=str(Path(__file__).resolve().parents[2] / 'test/吸烟.mp4'))
    ap.add_argument('--output', required=True)
    ap.add_argument('--rounds', type=int, default=3)
    ap.add_argument('--frames', type=int, default=12)
    args = ap.parse_args()
    cfg, bs = load_settings(), load_behaviors()
    state = requests.get(args.ui + '/api/state', timeout=10).json() if not args.url else {}
    url = args.url or state['service']['local_url']
    # Images are encoded in memory; this benchmark does not create frame files.
    cap = cv2.VideoCapture(args.video)
    frames, labels = [], []
    try:
        for i in range(args.frames):
            t = 2 + 6 * i / max(1, args.frames - 1)
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ok, image = cap.read()
            if not ok:
                raise RuntimeError(f'Cannot decode video frame at {t:.2f}s')
            h, w = image.shape[:2]
            side = int(cfg['extract']['max_side'])
            if max(h, w) > side:
                ratio = side / max(h, w)
                image = cv2.resize(image, (round(w * ratio), round(h * ratio)))
            ok, encoded = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 82])
            if not ok:
                raise RuntimeError('JPEG encoding failed')
            frames.append(base64.b64encode(encoded.tobytes()).decode('ascii'))
            labels.append(f't={t:.1f}s')
    finally:
        cap.release()
    budget = generation_budget(bs, with_alert=True, with_actions=True, with_risks=True)

    def infer(images, timestamps, **kw):
        payload = dict(kw, frames=[{'b64': f} for f in images], labels=timestamps,
                       few_shot=bool(bs.defaults.get('few_shot', True)))
        payload['behaviors'] = behavior_payload(bs)
        response = requests.post(url + '/infer', json=payload, timeout=120)
        response.raise_for_status()
        return response.json()

    def window():
        start = time.perf_counter()
        result = infer_with_review(infer, frames, labels, bs, max_new_tokens=budget,
                                  with_alert=True, with_actions=True, with_risks=True)
        return {'wall_sec': time.perf_counter()-start, **{k:result.get(k) for k in (
            'latency_sec','tokens_out','tokens_prompt','batch','finish_reason','verification',
            'error','parsed','timing','generation_retry')}}

    rows = []
    for index in range(args.rounds):
        start = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda _: window(), range(2)))
        row = {'round':index+1, 'wall_sec':time.perf_counter()-start, 'results':results}
        rows.append(row)
        print(json.dumps({'round':index+1,'wall_sec':round(row['wall_sec'],3),
                         'tokens':[r['tokens_out'] for r in results],
                         'batches':[r['batch'] for r in results],
                         'errors':[r['error'] for r in results]},ensure_ascii=True),flush=True)
    coverage = 2 * float(cfg['live']['window_stride_sec'])
    median = statistics.median(r['wall_sec'] for r in rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps({'rows':rows,'median_pair_sec':median,
        'coverage_sec':coverage,'budget':budget,'frames':args.frames,'video':args.video,
        'live':cfg['live'],'passes':median<=coverage and all(not r['error'] for row in rows for r in row['results'])},
        ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'Median pair: {median:.3f}s; coverage: {coverage:.3f}s; target: median <= coverage')
    return 0 if median<=coverage and all(not r['error'] for row in rows for r in row['results']) else 1


if __name__ == '__main__':
    raise SystemExit(main())
