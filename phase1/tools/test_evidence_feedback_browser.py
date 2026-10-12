"""Real local API + browser review workflow in temporary storage, no GPU/camera."""
import copy
import json
from pathlib import Path
import sys
import subprocess
import tempfile
import types
from unittest.mock import patch

import cv2
import numpy as np
from fastapi.testclient import TestClient
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from webui import server as S
from civility.extract import find_ffmpeg

OUT = ROOT / 'runs/_workspace_20261012'


def main():
    OUT.mkdir(exist_ok=True)
    saved = S.STATE.cfg, S.STATE.jobs, S.STATE.uploads, S.STATE.session
    errors = []
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        S.STATE.cfg = copy.deepcopy(S.STATE.cfg)
        S.STATE.cfg['paths'].update({key: str(root/key) for key in ('data_dir', 'frames_dir', 'results_dir')})
        S.STATE.jobs, S.STATE.uploads = {}, {}
        S.STATE.session = types.SimpleNamespace(state='ready', started_at=None, status=lambda: {'state': 'ready'})
        library = S.video_library()
        library.uploads_root.mkdir(parents=True)
        source = library.uploads_root / 'fixture.mp4'
        writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*'mp4v'), 10, (480, 270))
        for i in range(80):
            frame = np.full((270, 480, 3), 230, dtype=np.uint8)
            cv2.putText(frame, f'Frame {i} / {i/10:.1f}s', (35, 130), cv2.FONT_HERSHEY_SIMPLEX, .8, (140, 95, 60), 2)
            writer.write(frame)
        writer.release()
        original = source.with_name('original.mp4')
        source.rename(original)
        subprocess.run([find_ffmpeg()[0], '-hide_banner', '-loglevel', 'error', '-i', str(original),
                        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(source)], check=True, timeout=30)
        job = S.JobRecord('fixture', '证据联动测试.mp4', state='done', windows_seen=3)
        job._archive_source = source
        job.alerts = [{'behavior_id': 'littering', 'behavior_name': '乱扔垃圾', 'severity': 'medium', 'window_ids': ['w1'], 't_start': 2, 't_end': 6, 'reason': '动作过程仅用于界面测试'}]
        library.begin(job)
        (library.frames_root/'fixture').mkdir(parents=True)
        for i in range(3):
            indices = [i*20, i*20+10]
            library.record(job, {'window_id': f'w{i}', 't_start': i*2, 't_end': i*2+4, 'frames': 2,
                                'frame_indices': indices, 'frame_labels': [f't={n/10:.1f}s' for n in indices],
                                'frame_times': [n/10 for n in indices], 'scene': '<script>不可执行</script> 复核窗口',
                                'violations': {'littering': {'violation': i==1, 'confidence': .95, 'observable': '测试可见证据', 'alert': i==1}}})
            for n in indices:
                cv2.imwrite(str(library.frames_root/'fixture'/f'f_{n:05d}.jpg'), frame)
        library.save(job)
        try:
            with patch.object(S, 'ensure_current_runtime'), patch.object(S, 'runtime_status', return_value={'protocol':3,'restart_required':False}), TestClient(S.create_app()) as client, sync_playwright() as p:
                browser = p.chromium.launch(channel='msedge', headless=True)
                page = browser.new_page(viewport={'width':1440,'height':1000}, reduced_motion='reduce')
                page.on('pageerror', lambda error: errors.append(str(error)))
                page.add_init_script("window.EventSource=class {constructor(){this.readyState=1;}close(){}};")
                def route(request):
                    req = request.request
                    path = req.url.split('http://review.test', 1)[-1]
                    headers = {k:v for k,v in req.headers.items() if k.lower() in ('content-type','range')}
                    response = client.request(req.method, path, content=req.post_data_buffer, headers=headers)
                    returned = {k:v for k,v in response.headers.items() if k.lower() not in ('content-length','content-encoding')}
                    request.fulfill(status=response.status_code, headers=returned, body=response.content)
                page.route('**/*', route)
                page.goto('http://review.test/', wait_until='networkidle')
                page.wait_for_function('!restoringJobs&&!historyLoading&&currentJob==="fixture"')
                page.locator('aside [data-tab="detect"]').click()
                page.locator('#alerts button').click()
                page.locator('#evidence-grid img').first.wait_for()
                assert page.locator('#evidence-grid figure').count()==6
                assert page.locator('#evidence-dialog script').count()==0
                assert page.locator('#feedback-behavior').input_value()=='littering'
                page.locator('#evidence-player').wait_for()
                page.wait_for_function('document.querySelector("#evidence-player").readyState>=1')
                page.locator('[data-evidence-seek="3"]').click()
                page.wait_for_function('Math.abs(document.querySelector("#evidence-player").currentTime-3)<.2')
                page.locator('#feedback-frame').select_option('f_30')
                page.locator('[data-label="false_positive"]').click()
                page.locator('#feedback-note').fill('手中物品为指甲剪 <script>，并非烟')
                page.locator('#feedback-save').click()
                page.wait_for_function('document.querySelector("#feedback-status").textContent.includes("已保存")')
                assert client.get('/api/feedback/fixture').json()['feedback'][0]['label']=='false_positive'
                assert library.windows('fixture')[1]['violations']['littering']['violation'] is True
                page.screenshot(path=str(OUT/'evidence-feedback-desktop.png'))
                page.locator('#evidence-close').click()
                page.locator('#alerts button').click()
                page.wait_for_function('document.querySelector("#feedback-note").value.includes("指甲剪")')
                assert page.locator('#feedback-labels .selected').inner_text()=='误报'
                with page.expect_download() as download: page.locator('#evidence-grid a').first.click()
                assert download.value.suggested_filename.endswith('.jpg')
                with page.expect_download() as download: page.locator('#feedback-export').click()
                assert download.value.suggested_filename=='fixture-feedback.json'
                page.set_viewport_size({'width':390,'height':844})
                assert page.locator('#evidence-dialog').evaluate('e=>e.scrollWidth<=e.clientWidth+2')
                page.screenshot(path=str(OUT/'evidence-feedback-mobile.png'))
                page.locator('#evidence-locate').click()
                assert page.locator('#windows .evidence-focus').get_attribute('data-window-id')=='w1'
                page.set_viewport_size({'width':1440,'height':1000})
                # A missed action can be reviewed from a window without an alert.
                page.locator('#windows .act').filter(has=page.locator('.t', has_text='#1')).locator('button', has_text='标记漏报').click()
                page.locator('#feedback-behavior').select_option('smoking')
                page.locator('[data-label="missed"]').click();page.locator('#feedback-save').click()
                page.wait_for_function('document.querySelector("#feedback-status").textContent.includes("已保存")')
                page.locator('#evidence-close').click()
                # Upload reservation survives browser reload, without an SSE worker.
                page.locator('#file').set_input_files({'name':'pending.mp4','mimeType':'video/mp4','buffer':source.read_bytes()})
                page.wait_for_function('uploadedJob&&!uploading')
                pending_id = page.evaluate('uploadedJob')
                page.reload(wait_until='networkidle')
                page.wait_for_function('!restoringJobs&&!historyLoading')
                assert page.evaluate('uploadedJob')==pending_id and page.evaluate('fileRunning') is False
                assert not page.locator('#btn-detect').is_disabled()
                assert not S._active_job()
                browser.close()
                assert not errors, errors
                (OUT/'evidence-feedback-browser.json').write_text(json.dumps({'passed':True,'page_errors':errors,'checks':['evidence neighbors','video seek','feedback persistence','XSS escaping','screenshot download','feedback export','mobile overflow','window locate','missed feedback','pending upload reload']}), encoding='utf-8')
                print('PASS real API + browser evidence, feedback, downloads, mobile layout and upload recovery')
        finally:
            S.STATE.cfg, S.STATE.jobs, S.STATE.uploads, S.STATE.session = saved


if __name__ == '__main__': main()
