"""Complete preset / ROI / statistics / evidence / QA browser workflow in isolation."""
import json
from pathlib import Path
import sys
import time
from playwright.sync_api import sync_playwright
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tests'))
from test_workstation_api import WorkstationApiTests

OUT=ROOT/'runs/_workstations_20261012'


def main():
    OUT.mkdir(exist_ok=True);fixture=WorkstationApiTests();fixture.setUp();errors=[]
    try:
        with sync_playwright() as p:
            browser=p.chromium.launch(channel='msedge',headless=True)
            page=browser.new_page(viewport={'width':1440,'height':1100},reduced_motion='reduce')
            page.on('pageerror',lambda error:errors.append(str(error)))
            def route(handler):
                request=handler.request;path=request.url.split('http://stations.test',1)[-1]
                headers={k:v for k,v in request.headers.items() if k in ('content-type','range')}
                response=fixture.client.request(request.method,path,content=request.post_data_buffer,headers=headers)
                handler.fulfill(status=response.status_code,headers={k:v for k,v in response.headers.items() if k not in ('content-length','content-encoding')},body=response.content)
            page.route('**/*',route);page.goto('http://stations.test/',wait_until='networkidle')
            page.wait_for_function('stationsUI?.available&&!runtimeOutdated&&!restoringJobs')
            assert page.locator('#m-profile option').count()==3 and page.locator('#f-profile option').count()==3
            page.locator('aside [data-tab="behaviors"]').click();page.locator('#profile-editor').select_option('workstations')
            page.wait_for_function('document.querySelector("#yaml-box").value.includes("station_away")')
            page.locator('#profile-copy-name').fill('办公室预设');page.locator('#profile-copy').click()
            page.wait_for_function('document.querySelector("#profile-editor").value.startsWith("custom-")')
            custom=page.locator('#profile-editor').input_value()
            assert '办公室预设' in page.locator('#profile-editor').inner_text()
            page.locator('#profile-default').click()
            assert fixture.client.get('/api/profiles').json()['default_id']==custom
            page.locator('aside [data-tab="detect"]').click()
            page.locator('#f-profile').select_option('workstations');page.locator('#f-mock').check()
            page.locator('#file').set_input_files({'name':'walking.mp4','mimeType':'video/mp4','buffer':fixture.video.read_bytes()})
            page.wait_for_function('uploadedJob&&!uploading');jid=page.evaluate('uploadedJob')
            assert page.locator('#btn-detect').is_disabled()
            page.locator('#f-station-setup').click();page.wait_for_function('stationsUI.image&&stationsUI.draft')
            rectangle=page.locator('#station-canvas').bounding_box()
            for x,y in [(.01,.01),(.5,.01),(.5,.99),(.01,.99)]:page.locator('#station-canvas').click(position={'x':x*rectangle['width'],'y':y*rectangle['height']})
            page.locator('#station-name').fill('1号工位');page.locator('#station-add').click()
            page.locator('#station-away').fill('2');page.locator('#station-empty').fill('2');page.locator('#station-confirm').fill('1')
            page.locator('#station-save').click();page.wait_for_function('!document.querySelector("#station-setup-dialog").open')
            assert not page.locator('#btn-detect').is_disabled()
            page.locator('#btn-detect').click()
            page.wait_for_function('!fileRunning&&document.querySelector("#f-station-people tr")')
            page.wait_for_function('stationsUI.reports.file?.people?.[0]?.away_count===1')
            report=fixture.client.get('/api/stations/'+jid).json();away=report['people'][0]['away_sec']
            assert 6<=away<=8,report['people']
            assert f'{away:.1f}秒' in page.locator('#f-station-people').inner_text()
            assert report['profile']['id']=='workstations'
            for thumb in page.locator('#windows .thumb').all():thumb.scroll_into_view_if_needed()
            page.wait_for_function('Array.from(document.querySelectorAll("#windows .thumb")).every(img=>img.complete&&img.naturalWidth>0)')
            assert '—s' not in page.locator('#windows').inner_text()
            assert '工位复核详情' in page.locator('#windows').inner_text()
            page.locator('#f-station-timeline .away').first.click();page.locator('#evidence-grid img').first.wait_for()
            assert page.locator('#evidence-grid img').count()>=2
            page.locator('[data-label="false_positive"]').click();page.locator('#feedback-note').fill('测试人工复核');page.locator('#feedback-save').click()
            page.wait_for_function('document.querySelector("#feedback-status").textContent.includes("已保存")');page.locator('#evidence-close').click()
            page.locator('#f-station-people [data-person]').first.click();page.locator('#station-person-name').fill('员工A');page.locator('#station-person-save').click()
            page.wait_for_function('document.querySelector("#f-station-people").textContent.includes("员工A")')
            page.wait_for_function('document.querySelector("#qa-video").value')
            page.locator('#qa-question').fill('员工A累计离岗多少秒，几次？');page.locator('#qa-ask').click();page.locator('#qa-answers .qa-answer').wait_for()
            assert f'{away:.1f}秒' in page.locator('#qa-answers').inner_text()
            with page.expect_download() as download:page.locator('#f-station-export').click()
            assert download.value.suggested_filename.endswith('-workstations.json')
            page.evaluate('scrollTo(0,0)');page.screenshot(path=str(OUT/'station-desktop.png'),full_page=True)
            for width in (390,768,1440):
                page.set_viewport_size({'width':width,'height':900})
                assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+1'),width
            page.set_viewport_size({'width':390,'height':844});page.evaluate('scrollTo(0,0)');page.screenshot(path=str(OUT/'station-mobile.png'),full_page=True)
            page.reload(wait_until='networkidle');page.wait_for_function('!restoringJobs&&!historyLoading&&stationsUI.reports.file?.people?.length')
            assert '员工A' in page.locator('#f-station-people').inner_text()
            browser.close();assert not errors,errors
            (OUT/'browser.json').write_text(json.dumps({'passed':True,'page_errors':errors,'checks':['presets','copy/default','ROI polygon','complete analysis','durations','event evidence','feedback','person name','QA','export','responsive','history reload']},ensure_ascii=False),encoding='utf-8')
            print('PASS workstation presets, ROI, statistics, evidence, feedback, QA and history across desktop/mobile')
    finally:fixture.tearDown()


if __name__=='__main__':main()
