"""Reproduce a late stopping poll overwriting an already-completed release."""
import argparse
import json
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/_workspace_20261012'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--html',type=Path)
    args=parser.parse_args();source=(args.html or ROOT/'webui/static/index.html').read_text(encoding='utf-8')
    ready={'state':'ready','message':'模型就绪','jobid':'fixture-job','node':'fixture-node'}
    stopped={'state':'stopped','message':'已停止并释放计算节点','jobid':None,'node':None,'release_pending':False,'start_worker_alive':False}
    state=ready.copy();hold_next=False;held=None;starts=0;stops=0;stop_fail=False;errors=[]
    def payload(service):
        return {'runtime':{'protocol':3,'restart_required':False},'service':service,'jobs':[],
                'behaviors':{'enabled':[]},'aggregate':{'mode':'ai'},'orphans':[]}
    with sync_playwright() as p:
        browser=p.chromium.launch(channel='msedge',headless=True)
        page=browser.new_page(viewport={'width':1440,'height':1000},reduced_motion='reduce')
        page.on('pageerror',lambda error:errors.append(str(error)))
        page.add_init_script("window.speechSynthesis={getVoices:()=>[],cancel:()=>{}};")
        def respond(route):
            nonlocal state,hold_next,held,starts,stops
            url=route.request.url
            if url.endswith('/static/alert_speech.js'):
                return route.fulfill(content_type='text/javascript',body=(ROOT/'webui/static/alert_speech.js').read_text(encoding='utf-8'))
            if url.endswith('/api/state'):
                if hold_next:hold_next=False;held=route;return
                result=payload(state)
            elif url.endswith('/api/service/stop'):
                stops+=1
                if stop_fail:
                    state={'state':'error','message':'释放失败，请重试停止','release_pending':True,'jobid':'fixture-job'}
                    return route.fulfill(status=503,content_type='application/json',body=json.dumps({'detail':state['message']}))
                state=stopped.copy();result={'ok':True,'message':state['message'],'service':state}
            elif url.endswith('/api/service/start'):
                starts+=1;state={'state':'pending','message':'排队中','start_worker_alive':True}
                result={'ok':True,'message':'启动已提交','service':state}
            elif url.endswith('/api/videos'):result={'videos':[]}
            elif '/api/' in url:result={}
            else:return route.fulfill(content_type='text/html',body=source)
            route.fulfill(content_type='application/json',body=json.dumps(result))
        page.route('**/*',respond);page.goto('http://release.test/',wait_until='networkidle')
        page.wait_for_function('!runtimeOutdated&&!restoringJobs&&!refreshBusy')
        page.locator('aside button[data-tab="service"]').click()
        hold_next=True
        with page.expect_request(lambda r:r.url.endswith('/api/state')):
            page.evaluate('void refresh()')
        page.wait_for_function('refreshBusy')
        assert held is not None
        page.locator('#btn-stop').click()
        page.wait_for_function('document.querySelector("#flashes").textContent.includes("已停止并释放")')
        # This response describes an earlier stopping snapshot, but arrives
        # after the stop response. It must not overwrite the completed release.
        old={'state':'stopping','message':'已停止并释放计算节点','node':'fixture-node','jobid':'fixture-job'}
        held.fulfill(content_type='application/json',body=json.dumps(payload(old)))
        page.wait_for_function('!refreshBusy')
        assert page.locator('#svc-bar').evaluate('e=>e.style.width')=='0%', 'released progress did not reset'
        assert '已停止' in page.locator('#c-svc').inner_text(), page.locator('#c-svc').inner_text()
        assert 'fixture-node' not in page.locator('#c-svc').inner_text()
        assert page.locator('#btn-start').is_enabled(), 'released service cannot restart'
        assert page.locator('#btn-stop').is_disabled()
        page.locator('#btn-start').click()
        page.wait_for_function('document.querySelector("#c-svc").textContent.includes("排队中")')
        assert starts==1
        assert page.locator('#btn-start').is_disabled()
        state=ready.copy();page.evaluate('void refresh()')
        page.wait_for_function('document.querySelector("#c-svc").textContent.includes("模型就绪")')
        stop_fail=True;page.locator('#btn-stop').click()
        page.wait_for_function('document.querySelector("#c-svc").textContent.includes("释放失败")')
        assert page.locator('#btn-stop').is_enabled()
        assert page.locator('#btn-start').is_disabled()
        stop_fail=False;page.locator('#btn-stop').click()
        page.wait_for_function('document.querySelector("#c-svc").textContent.includes("已停止")')
        assert page.locator('#btn-start').is_enabled()
        state={**stopped,'start_worker_alive':True};page.evaluate('void refresh()')
        page.wait_for_function('document.querySelector("#c-svc").textContent.includes("收尾")')
        assert page.locator('#btn-start').is_disabled()
        assert '计算节点已释放' in page.locator('#svc-msg').inner_text()
        state=stopped.copy();page.evaluate('void refresh()')
        page.wait_for_function('document.querySelector("#btn-start").disabled===false')
        assert page.locator('#svc-bar').evaluate('e=>e.style.width')=='0%'
        assert not errors,errors
        OUT.mkdir(exist_ok=True);page.locator('#flashes').evaluate('e=>e.replaceChildren()')
        page.screenshot(path=str(OUT/'service-stopped.png'),full_page=True)
        (OUT/'service-stop-browser.json').write_text(json.dumps({'status':'passed','late_poll_ignored':True,'progress_reset':True,'restart_enabled':True,'retry_after_release_failure':True,'startup_cleanup_visible':True,'starts':starts,'stops':stops,'page_errors':errors},indent=2),encoding='utf-8')
        browser.close();print('PASS release progress/header/restart, stale poll, release retry and startup cleanup')


if __name__=='__main__':main()
