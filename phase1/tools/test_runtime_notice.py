"""Browser regression: stale server detection and model review display, offline."""
import json
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/test-video-validation/followup'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    state = {'service': {'state': 'ready', 'model': 'Qwen3-VL'}, 'aggregate': {'mode': 'ai'},
             'behaviors': {'enabled': []}, 'extract': {'fps': 1, 'max_frames_per_window': 4},
             'live': {}, 'cost': {}, 'log': [], 'orphans': [], 'jobs': []}
    with sync_playwright() as p:
        browser = p.chromium.launch(channel='msedge', headless=True)
        page = browser.new_page(viewport={'width': 1365, 'height': 960})
        errors = []
        page.on('pageerror', lambda error: errors.append(str(error)))
        def respond(route):
            if '/static/alert_speech.js' in route.request.url:
                route.fulfill(status=200, content_type='text/javascript', body=(ROOT / 'webui/static/alert_speech.js').read_text(encoding='utf-8'))
            elif '/api/state' in route.request.url:
                route.fulfill(status=200, content_type='application/json', body=json.dumps(state))
            elif '/api/' in route.request.url:
                route.fulfill(status=200, content_type='application/json', body='{}')
            else:
                route.fulfill(status=200, content_type='text/html',
                              body=(ROOT / 'webui/static/index.html').read_text(encoding='utf-8'))
        page.route('**/*', respond)
        page.goto('http://runtime.test/', wait_until='networkidle')
        page.wait_for_function('runtimeOutdated && !refreshBusy && !restoringJobs')
        assert page.locator('#runtime-note').is_visible()
        page.evaluate("uploadedJob='fixture'; updateControls()")
        assert page.locator('#btn-detect').is_disabled()
        assert page.locator('#m-start').is_disabled()
        assert page.locator('#btn-start').is_disabled()
        page.locator('aside button[data-tab="detect"]').click()
        page.screenshot(path=str(OUT / 'stale-runtime-notice.png'), full_page=True)
        print('PASS old server has visible restart instructions and cannot start a new analysis')

        state['runtime'] = {'protocol': 3, 'restart_required': False}
        state['extract'] = {'fps': 3, 'max_frames_per_window': 18}
        page.evaluate('refresh()')
        page.wait_for_function('!runtimeOutdated && !refreshBusy')
        assert not page.locator('#runtime-note').is_visible()
        assert page.locator('#btn-detect').is_enabled()
        state['service']['state'] = 'stopping'
        page.evaluate('refresh()')
        page.wait_for_function('!serviceReady && !refreshBusy')
        assert '正在停止并释放节点' in page.locator('#c-svc').text_content()
        assert '正在停止并释放节点' in page.locator('#svc-msg').text_content()
        assert '正在停止并释放节点' in page.locator('#svc-detail').text_content()
        assert page.locator('#btn-start').is_disabled()
        state['service']['state'] = 'ready'
        page.evaluate('refresh()')
        page.wait_for_function('serviceReady && !refreshBusy')
        print('PASS stopping is shown in Chinese and cannot launch another service')
        state['runtime']['restart_required'] = True
        page.evaluate('refresh()')
        page.wait_for_function('runtimeOutdated && !refreshBusy')
        assert page.locator('#btn-detect').is_disabled()
        print('PASS current server resumes analysis; changed source/settings block it again')

        page.evaluate('''() => {
          const card=windowCard({window_id:'test',frames:1,t_start:0,t_end:4,scene:'持杯→脱手→落地',
            hits:[{name:'乱扔垃圾',confidence:.95,observable:'杯从手中脱离并落到草地'}],
            verification:{status:'confirmed',raw:'<img src=x onerror=alert(1)>核验证据完整'}}, 'test');
          document.querySelector('#windows').replaceChildren(card);
        }''')
        assert '已由模型二次核验确认' in page.locator('#windows').text_content()
        page.locator('#windows details summary').click()
        assert page.locator('#windows details img').count() == 0
        assert '<img src=x onerror=alert(1)>' in page.locator('#windows pre').text_content()
        assert not errors, errors
        print('PASS automatic model review is visible and raw output is safely escaped')
        immediate=page.evaluate('''() => {
          clearMonitorHistory();
          document.querySelector('#m-feed').replaceChildren();
          for(let i=1;i<=2;i++){
            enqueueWindow({window_id:'latency-'+i,index:i,cumulative:i,t_start:(i-1)*2,
              t_end:i*2,scene:'正常活动',people:[],hits:[],risks:[],frames:4,
              frame0:i*4,latency:3.2,roundtrip_sec:3.4,result_age_sec:3.5},'latency');
          }
          return {cards:document.querySelectorAll('#m-feed .act').length,
            latest:document.querySelector('#mm-win').textContent};
        }''')
        assert immediate=={'cards':2,'latest':'2'},immediate
        assert '立即显示' in page.locator('#m-pace').text_content()
        assert not errors,errors
        print('PASS both completed windows are visible immediately without a pacing queue')
        browser.close()


if __name__ == '__main__':
    main()
