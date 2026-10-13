"""Offline integration of unified video workflows; no camera, GPU or real deletion."""
import io
import json
from pathlib import Path
import wave
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/_workspace_20261012'


def main():
    OUT.mkdir(exist_ok=True)
    errors, writes = [], []
    videos = [{'video_id': name, 'filename': name+'.mp4', 'windows_seen': 2,
               'archived_windows': 2, 'source_available': False, 'analysis_revision': 'original'}
              for name in ['fixture', 'second']]
    alerts = [{'behavior_id': 'littering', 'behavior_name': '乱扔垃圾', 'severity': 'medium',
               't_start': 0, 't_end': 4, 'window_ids': ['w0'], 'reason': '丢弃物品后离开'}]
    record = {'window_id': 'w0', 'index': 1, 't_start': 0, 't_end': 4, 'frames': 2,
              'frame_indices': [0, 6], 'frame_labels': ['t=0.0s', 't=2.0s'],
              'archive_revision': 'original', 'scene': '人在公园丢弃纸杯',
              'hits': [{'behavior_id': 'littering', 'name': '乱扔垃圾', 'confidence': .95,
                        'observable': '物品落地后人物离开', 'alert': True}]}
    pending = {**record, 'window_id': 'w1', 'index': 2, 'hits': [],
               'evidence_warnings': [{'name': '吸烟', 'issues': ['不能确认手中物品']}], 'risks': []}
    wav = io.BytesIO()
    with wave.open(wav, 'wb') as audio:
        audio.setnchannels(1); audio.setsampwidth(2); audio.setframerate(24000); audio.writeframes(b'\0'*2400)
    speech_fail = False
    service_state = 'ready'
    cluster_ok = True
    with sync_playwright() as p:
        browser = p.chromium.launch(channel='msedge', headless=True)
        page = browser.new_page(viewport={'width':1440,'height':1000}, reduced_motion='reduce')
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.add_init_script("""window.spoken=[];window.spokenRates=[];window.audioPlayed=0;
          window.SpeechSynthesisUtterance=class {constructor(text){this.text=text;}};
          Object.defineProperty(window,'speechSynthesis',{value:{getVoices:()=>[{name:'系统女声',lang:'zh-CN',voiceURI:'system'}],
            addEventListener:()=>{},cancel:()=>{},speak:u=>{spoken.push(u.text);spokenRates.push(u.rate);setTimeout(()=>u.onend&&u.onend(),10);}}});
          window.Audio=class {play(){window.audioPlayed++;setTimeout(()=>this.onended&&this.onended(),20);return Promise.resolve();}pause(){}removeAttribute(){}};
          window.EventSource=class {static CLOSED=2;constructor(url){this.readyState=1;window.stream=this;}close(){}};
        """)
        def respond(route):
            url, method = route.request.url, route.request.method
            if '/static/' in url:
                return route.fulfill(content_type='text/javascript', body=(ROOT/'webui/static'/url.split('?')[0].rsplit('/',1)[-1]).read_text(encoding='utf-8'))
            if '/frames/' in url:
                return route.fulfill(content_type='image/svg+xml', body='<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="720"><rect width="1280" height="720" fill="#496d96"/></svg>')
            if method not in ('GET', 'HEAD'): writes.append({'url':url,'method':method})
            if url.endswith('/api/state'):
                data={'runtime':{'protocol':3,'restart_required':False}, 'service':{'state':service_state},
                      'jobs':[], 'behaviors':{'enabled':[]}, 'aggregate':{'mode':'ai'}}
            elif url.endswith('/api/videos'): data={'videos':videos}
            elif url.endswith('/analysis'):
                video_id=url.split('/')[-2]
                data={**next(v for v in videos if v['video_id']==video_id), 'alerts':alerts, 'window_results':[record,pending]}
            elif method=='DELETE' and '/api/videos/' in url:
                video_id=url.split('/')[-1];videos[:]=[v for v in videos if v['video_id']!=video_id]
                data={'ok':True,'message':'已删除视频'}
            elif url.endswith('/ask'):
                data={'answer':'<script>不可执行</script> 可见丢弃物品。', 'model':'已部署模型', 'sampled_frames':2,
                      'citations':[{'image_url':'/api/videos/fixture/frames/f_6','time_sec':2,'sample_number':7,'description':'物品落地'}]}
            elif url.endswith('/api/upload'): data={'job_id':'fresh','size_mb':1}
            elif '/api/detect/fresh' in url:
                videos.insert(0,{'video_id':'fresh','filename':'new.mp4','windows_seen':1,'source_available':False})
                data={'ok':True}
            elif '/api/speech/synthesize' in url:
                if speech_fail:return route.fulfill(status=503,content_type='application/json',body='{"detail":"unavailable"}')
                return route.fulfill(content_type='audio/wav',body=wav.getvalue())
            elif url.endswith('/api/cluster/check'): data={'ok':cluster_ok,'host':'fixture-host','error':'offline'}
            elif '/api/' in url: data={}
            else:return route.fulfill(content_type='text/html',body=(ROOT/'webui/static/index.html').read_text(encoding='utf-8'))
            route.fulfill(content_type='application/json',body=json.dumps(data))
        page.route('**/*',respond)
        page.goto('http://workspace.test/',wait_until='networkidle')
        page.wait_for_function('!runtimeOutdated&&!restoringJobs&&!historyLoading')
        page.locator('aside button[data-tab="detect"]').click()
        page.wait_for_function('!historyLoading&&document.querySelector("#windows .act")')
        assert page.locator('aside button[data-tab="qa"]').count()==0
        assert page.locator('#tab-detect #drop').is_visible() and page.locator('#tab-detect #qa-question').is_visible()
        assert page.locator('.act.alerted .alert-badge').inner_text()=='已告警'
        assert page.locator('.act.alerted .behavior-hit strong').evaluate('e=>getComputedStyle(e).color')=='rgb(168, 44, 37)'
        assert page.locator('.act:not(.alerted) .evidence-review').count()==1
        assert '/api/videos/' in page.locator('#windows .thumb').first.get_attribute('src')
        page.locator('#qa-question').fill('他在哪一帧乱丢垃圾？');page.locator('#qa-ask').click()
        page.locator('#qa-answers .qa-answer').wait_for()
        assert page.locator('#qa-answers script').count()==0
        assert '2.00 秒' in page.locator('#qa-answers').inner_text()
        page.locator('#qa-answers img').click();assert page.locator('#image-viewer').is_visible();page.keyboard.press('Escape')
        page.locator('#qa-delete').click();page.locator('#video-delete-cancel').click()
        assert len(videos)==2
        page.locator('#qa-delete').click();page.locator('#video-delete-confirm').click()
        page.wait_for_function('!qaDeleting&&!historyLoading&&document.querySelector("#qa-video").value==="second"')
        assert len(videos)==1 and not page.locator('#qa-answers').inner_text()
        page.locator('#file').set_input_files({'name':'new.mp4','mimeType':'video/mp4','buffer':b'fixture'})
        page.wait_for_function('uploadedJob==="fresh"&&!uploading')
        assert page.locator('#qa-video').input_value()==''
        page.locator('#btn-detect').click()
        page.wait_for_function('window.stream&&fileRunning')
        page.evaluate('r=>stream.onmessage({data:JSON.stringify({type:"window",...r,total:1,progress:1})})',record)
        page.evaluate('a=>stream.onmessage({data:JSON.stringify({type:"alerts",alerts:a})})',alerts)
        assert page.locator('#windows .alerted').count()==1
        page.evaluate('a=>stream.onmessage({data:JSON.stringify({type:"done",summary:{state:"done",alerts:a}})})',alerts)
        page.wait_for_function('!fileRunning&&document.querySelector("#qa-video").value==="fresh"')
        page.locator('#speech-rate').select_option('0.75');page.locator('#speech-test').click()
        page.wait_for_function('spoken.length>0&&!speech.active')
        assert page.evaluate('spokenRates.at(-1)')==.75
        assert '标准普通话' in page.locator('#speech-status').inner_text()
        assert page.evaluate('audioPlayed')==0  # Preview has no neural HTTP wait.
        speech_fail=True;page.locator('#speech-test').click()
        page.wait_for_function('spoken.length>1&&!speech.active')
        assert '标准普通话' in page.locator('#speech-status').inner_text()
        page.locator('aside button[data-tab="cluster"]').click();page.locator('#btn-cluster-check').click()
        page.wait_for_function('document.querySelector("#c-vpn").classList.contains("ok")')
        assert page.locator('#c-vpn .dot').evaluate('e=>getComputedStyle(e).backgroundColor')=='rgb(21, 115, 71)'
        service_state='running';page.evaluate('refresh()')
        page.wait_for_function('document.querySelector("#c-svc").textContent.includes("隧道就绪")')
        assert 'ok' in page.locator('#c-svc').get_attribute('class')
        cluster_ok=False;page.locator('#btn-cluster-check').click()
        page.wait_for_function('document.querySelector("#c-vpn").classList.contains("err")')
        service_state='ready';cluster_ok=True;page.evaluate('refresh()')
        page.locator('#btn-cluster-check').click()
        for width,height in [(1440,1000),(1920,1080),(760,1000),(390,844)]:
            page.set_viewport_size({'width':width,'height':height})
            for tab in ['monitor','detect','service','behaviors','cluster']:
                page.locator(f'aside button[data-tab="{tab}"]').click()
                assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+1'),(width,tab)
            page.locator('aside button[data-tab="detect"]').click()
            if width in (1440,390):page.screenshot(path=str(OUT/f'workspace-{width}.png'),full_page=True)
        assert not errors,errors
        (OUT/'browser.json').write_text(json.dumps({'status':'passed','checks':['upload-analysis-auto-select','history-details','delete-cancel-and-confirm','citation-viewer-XSS','standard-Mandarin-preview-without-network-wait','green-connected-and-tunnel','red-alert-and-review-distinction','20-responsive-layouts'],'writes':writes},indent=2),encoding='utf-8')
        browser.close()
        print('PASS unified video workspace, deletion, speech fallback, semantic colors and responsive layouts')


if __name__=='__main__':main()
