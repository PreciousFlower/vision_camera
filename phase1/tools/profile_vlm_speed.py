"""Run controlled generation benchmarks inside an isolated GPU allocation."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cluster import vlm_service as V
from civility.recognition import generation_budget


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pkg', required=True)
    ap.add_argument('--cases', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--cache-only', action='store_true')
    ap.add_argument('--validate-only', action='store_true')
    args=ap.parse_args()
    import torch
    from transformers import CompileConfig
    V.STATE['behaviors']=V.load_behaviors_from(str(Path(args.pkg)/'behaviors.json'))
    V.load_model('Qwen/Qwen3-VL-4B-Instruct')
    if not V.STATE['ready']:
        raise RuntimeError(V.STATE['error'])
    bs=V.STATE['behaviors']
    budget=generation_budget(bs,with_alert=True,with_actions=True,with_risks=True)
    cases=json.loads(Path(args.cases).read_text(encoding='utf-8'))
    if args.validate_only:
        from civility.config import parse_behaviors
        from civility.recognition import infer_with_review
        results=[]
        def send(frames, labels, **options):
            definitions=parse_behaviors(options['behaviors'])
            item=V._Item(frames,labels,options['max_new_tokens'],options['few_shot'],
                         options['with_alert'],options['with_actions'],concurrent.futures.Future(),
                         options['with_risks'],definitions)
            return V.infer_batch([item])[0]
        for case in cases:
            result=infer_with_review(send,case['frames'],case['labels'],bs,max_new_tokens=budget,
                                    with_alert=True,with_actions=True,with_risks=True)
            hits=[k for k,v in (result.get('parsed') or {}).get('violations',{}).items() if v['violation']]
            results.append({**case,**result,'accepted_hits':hits})
            print(json.dumps({'video':case['video'],'start':case['t_start'],'expected':case['expected'],
                              'hits':hits,'error':result.get('error'),'seconds':result.get('latency_sec')},ensure_ascii=True),flush=True)
            Path(args.output).write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
        for video in {row['video'] for row in results}:
            rows=[row for row in results if row['video']==video]
            if not any(row['expected'] in row['accepted_hits'] for row in rows if row['expected']!='negative'):
                raise RuntimeError('No target detected: '+video)
        for row in results:
            if row.get('error') or (row['expected']=='negative' and row['accepted_hits']):
                raise RuntimeError('Recognition regression: '+row['window_id'])
            if row['expected'] in row['accepted_hits'] and not any(p.get('changed') for p in row['parsed'].get('people',[])):
                raise RuntimeError('Missing action process: '+row['window_id'])
        print('QUALITY_PASS',flush=True)
        print('PROFILE_COMPLETE',flush=True)
        return
    case=next(c for c in cases if c['expected']=='smoking' and c['t_start']==2)
    frame_indices=[round(i*(len(case['frames'])-1)/11) for i in range(12)]
    frames=[case['frames'][i] for i in frame_indices]
    labels=[case['labels'][i] for i in frame_indices]
    items=[V._Item(frames,labels,budget,True,True,True,concurrent.futures.Future(),True,bs) for _ in range(2)]
    # Match input geometry across experiments; only the execution strategy changes.
    rows=[]
    def measure(name, options):
        started=time.perf_counter()
        inputs=V._build_inputs(items).to(V.STATE['model'].device)
        if V.STATE.get('vision_cache'):
            from cluster.vision_cache import frame_key
            V.STATE['vision_cache'].keys=[frame_key(f) for it in items for f in it.frames]
        torch.cuda.synchronize()
        prepared=time.perf_counter()
        with torch.inference_mode():
            gen=V.STATE['model'].generate(**inputs,max_new_tokens=budget,do_sample=False,**options)
        torch.cuda.synchronize()
        generated=time.perf_counter()
        n=int(inputs.input_ids.shape[1])
        raws=V.STATE['processor'].batch_decode(gen[:,n:],skip_special_tokens=True)
        row={'variant':name,'prepare_sec':prepared-started,'generate_sec':generated-prepared,
             'total_sec':generated-started,'tokens_prompt':n,'tokens_out':int(gen.shape[1])-n,'raw':raws,
             'threads':torch.get_num_threads()}
        rows.append(row)
        print(json.dumps({k:v for k,v in row.items() if k!='raw'},ensure_ascii=True),flush=True)
        Path(args.output).write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Thread baseline:',torch.get_num_threads(),flush=True)
    if args.cache_only:
        cache=V.STATE.get('vision_cache')
        if cache is None:raise RuntimeError('No image feature cache')
        cache.supported=False
        for i in range(2):measure('uncached',{})
        cache.supported=True
        for i in range(3):measure('cached',{})
        from civility.prompts import extract_json,sanitize
        for row in rows:
            parsed=sanitize(extract_json(row['raw'][0]),bs,frame_count=len(frames))
            row['accepted_hits']=[k for k,v in parsed['violations'].items() if v['violation']]
            if 'smoking' not in row['accepted_hits']:raise RuntimeError('Positive smoking regression')
        Path(args.output).write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
        print('CACHE_STATS',cache.hits,cache.misses,cache.bytes,flush=True)
        print('PROFILE_COMPLETE',flush=True)
        return
    # Reproduce the original 7-CPU baseline independently of production defaults.
    if V.STATE.get('vision_cache'):
        V.STATE['vision_cache'].supported=False
    torch.set_num_threads(7)
    for i in range(2):measure('baseline',{})
    torch.set_num_threads(2)
    for i in range(2):measure('threads2',{})
    torch.set_num_threads(1)
    for i in range(2):measure('threads1',{})
    options={'cache_implementation':'static','compile_config':CompileConfig(mode='reduce-overhead',fullgraph=False)}
    for i in range(3):
        try: measure('static_compile',options)
        except Exception as exc:
            print('Compile experiment failed:',type(exc).__name__,str(exc)[:600],flush=True)
            break
    print('PROFILE_COMPLETE',flush=True)


if __name__=='__main__':main()
