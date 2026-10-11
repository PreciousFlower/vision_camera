"""Own a short profiling allocation, preserve any existing camera service."""
from pathlib import Path
import json
import shutil
import sys
import time
import argparse

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from check_validation_cluster import configure_auth
from civility.config import load_settings
from webui.cluster_client import ClusterClient


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--cache-only',action='store_true')
    parser.add_argument('--validate-only',action='store_true')
    args=parser.parse_args()
    configure_auth()
    root=Path(__file__).resolve().parents[1]
    cfg=load_settings()['cluster']
    output=root/('runs/latency-20261010/quality' if args.validate_only else
                 'runs/latency-20261010/cache-profile' if args.cache_only else 'runs/latency-20261010/profile')
    package=output/'package'
    package.mkdir(parents=True,exist_ok=True)
    remote='/share/home/u00009/apps/civility-speed-20261010'
    for directory in ('civility','cluster'):
        shutil.copytree(root/directory,package/directory,dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copy2(root/'tools/profile_vlm_speed.py',package/'profile_vlm_speed.py')
    prior=root/'runs/test-video-validation/model-followup-complete'
    shutil.copy2(prior/'package/behaviors.json',package/'behaviors.json')
    shutil.copy2(prior/'cases.json',package/'cases.json')
    script=f'''#!/bin/bash
#SBATCH --job-name=civility-speed-profile
#SBATCH --partition=A800
#SBATCH --gres=gpu:a800:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=48g
#SBATCH --time=00:15:00
#SBATCH --exclude={cfg['exclude_nodes']}
#SBATCH --output={remote}/profile-%j.log
export PYTHONPATH={remote}
export HF_HOME={cfg['remote_model_root']}/hf
export HF_HUB_CACHE={cfg['remote_model_root']}/hf/hub
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
cd {remote}
exec {cfg['remote_python']} profile_vlm_speed.py --pkg {remote} --cases {remote}/cases.json --output {remote}/profile.json {'--validate-only' if args.validate_only else '--cache-only' if args.cache_only else ''}
'''
    (package/'profile.sbatch').write_text(script,encoding='utf-8',newline='\n')
    client=ClusterClient(remote,cfg['remote_python'],cfg['remote_model_root'])
    errors=client.upload_tree(package,remote)
    if errors:raise RuntimeError(str(errors[:2]))
    job=None
    try:
        job=client.submit('profile.sbatch')
        print('Own profiling allocation:',job,flush=True)
        (output/'job.json').write_text(json.dumps({'job':job,'remote':remote}),encoding='utf-8')
        last=''
        deadline=time.monotonic()+830
        while time.monotonic()<deadline:
            code,log,_=client.run(f'tail -n 18 {remote}/profile-{job}.log 2>/dev/null',timeout=20)
            clean='\n'.join(line for line in log.splitlines() if line.startswith(('{','Thread','CACHE_STATS','PROFILE_COMPLETE','QUALITY_PASS','[load]','Compile experiment failed')))
            if clean!=last:
                print(clean,flush=True)
                last=clean
            state,node=client.job_state(job)
            if 'PROFILE_COMPLETE' in log or state in ('COMPLETED','FAILED','CANCELLED','TIMEOUT',''):
                break
            time.sleep(8)
        conn,sftp=client.sftp()
        try:
            sftp.get(remote+'/profile.json',str(output/'profile.json'))
            sftp.get(remote+f'/profile-{job}.log',str(output/'profile.log'))
        finally:sftp.close();conn.close()
    finally:
        if job:
            client.run(f'scancel {job}',timeout=20)
            print('Released own profiling allocation:',job,flush=True)


if __name__=='__main__':main()
