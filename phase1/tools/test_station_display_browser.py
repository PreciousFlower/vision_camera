"""Offline browser regression for workstation timeline noise and failed-window alerts."""
import json
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]


def main():
    failures = []
    with sync_playwright() as p:
        browser = p.chromium.launch(channel='msedge', headless=True)
        page = browser.new_page(viewport={'width': 1440, 'height': 1000})
        errors = []
        page.on('pageerror', lambda error: errors.append(str(error)))

        def route(handler):
            url = handler.request.url
            if '/static/' in url:
                return handler.fulfill(content_type='text/javascript', body=(ROOT / 'webui/static' / url.split('?')[0].rsplit('/', 1)[-1]).read_text(encoding='utf-8'))
            if '/api/state' in url:
                body = {'runtime': {'protocol': 3, 'restart_required': False}, 'service': {'state': 'ready'}, 'jobs': [], 'behaviors': {'enabled': []}, 'aggregate': {'mode': 'ai'}}
            elif '/api/' in url:
                body = {}
            else:
                return handler.fulfill(content_type='text/html', body=(ROOT / 'webui/static/index.html').read_text(encoding='utf-8'))
            handler.fulfill(content_type='application/json', body=json.dumps(body))

        page.route('**/*', route)
        page.goto('http://station-display.test/', wait_until='networkidle')
        page.wait_for_function('!runtimeOutdated&&!restoringJobs&&!historyLoading')
        page.locator('aside [data-tab="detect"]').click()
        page.evaluate('''() => {
          clearInterval(stationsUI.timer);
          stationsUI.jobs.file={id:'fixture',profile:{mode:'workstation'}};
          stationsUI.render('file', {time_sec:100,analysis_revision:'one',stations:[{id:'s1',name:'1号工位',expected_people:1,observed_count:1,resident_count:1}],
            people:[{person_id:'P001',name:'P001',state:'at_station',current_duration_sec:30,at_station_sec:40,away_sec:20,unknown_sec:40,away_count:1,longest_away_sec:20,unknown_reason:'检测短暂丢失',review_status:'confirmed'}],
            timeline:[{interval_id:'zero',person_id:'P001',station_id:'s1',state:'unknown',start:0,end:0},
              {interval_id:'a',person_id:'P001',station_id:'s1',state:'at_station',start:0,end:10},
              {interval_id:'b',person_id:'P001',station_id:'s1',state:'away',start:10,end:30},
              {interval_id:'c',person_id:'P001',station_id:'s1',state:'unknown',start:30,end:70,unknown_reason:'检测短暂丢失'},
              {interval_id:'d',person_id:'P001',station_id:'s1',state:'at_station',start:70,end:100}],
            station_timeline:[{interval_id:'szero',station_id:'s1',category:'normal',count:1,since:0,end:0},{interval_id:'snormal',station_id:'s1',category:'normal',count:1,since:70,end:100}]});
          const failed=windowCard({window_id:'failed',index:12,t_start:141.311999999998,t_end:178.546999999998,error:'Schema 校验失败：工位画面引用无效',station_review:{observations:[],stations:[]}},'fixture');
          document.querySelector('#windows').replaceChildren(failed);
          renderAlerts([{behavior_id:'station_away',behavior_name:'离岗超时',source:'workstation',t_start:150,t_end:178,window_ids:['failed'],interval_id:'b'}],false);
        }''')
        page.locator('#f-station-panel details summary').click()
        checks = {
            'zero/noisy intervals hidden by default': lambda: page.locator('#f-station-timeline .station-span').count() == 3,
            'failed inference cannot display alert badge': lambda: page.locator('#windows .alert-badge').count() == 0,
            'window timestamps rounded': lambda: '141.3s – 178.5s' in page.locator('#windows .top').inner_text(),
            'unknown has observable cause and explanation': lambda: '检测短暂丢失' in page.locator('#f-station-people').inner_text() and '不等于等待模型' in page.locator('#f-station-panel').inner_text(),
        }
        for name, check in checks.items():
            if check():
                print('PASS', name)
            else:
                failures.append(name)
                print('FAIL', name)
        if not failures:
            page.locator('#f-station-timeline-mode').select_option('all')
            assert page.locator('#f-station-timeline .station-span').count() == 5
            assert not any(line.startswith('0.0秒 ·') for line in page.locator('#f-station-timeline').inner_text().splitlines())
            page.locator('#f-station-timeline-kind').select_option('away')
            assert page.locator('#f-station-timeline .station-span').count() == 1
            page.locator('#f-station-timeline-kind').select_option('all')
            page.locator('#f-station-timeline-subject').select_option('station:s1')
            assert page.locator('#f-station-timeline .station-span').count() == 5
            page.evaluate('''() => {
              openEvidence=(...args)=>{window.lastEvidence=args;};
              currentJob='fixture';
              renderAlerts([{source:'workstation',review_status:'pending',provisional:true,behavior_id:'station_away',behavior_name:'离岗超时',duration_sec:20,threshold_time_sec:30,trigger_time_sec:30.2,t_start:10,t_end:30.2,interval_id:'b',window_ids:['failed']}],false);
            }''')
            assert '待 AI 复核' in page.locator('#alerts').inner_text()
            assert '大模型已确认' not in page.locator('#alerts').inner_text()
            assert '阈值时刻 30.0秒' in page.locator('#alerts').inner_text()
            page.locator('#alerts button').click()
            assert page.evaluate('lastEvidence[1]') == 'interval_b'
            page.evaluate('''() => {
              const report=stationsUI.reports.file;
              report.timeline=[];report.station_timeline=[];
              for(let i=0;i<30;i++)report.timeline.push({interval_id:'part'+i,person_id:'P001',station_id:'s1',state:i%2?'away':'at_station',start:i*2,end:i*2+2});
              stationsUI.render('file',report);
            }''')
            page.locator('#f-station-timeline-mode').select_option('meaningful')
            assert page.locator('#f-station-timeline .station-span').count() == 12
            page.locator('#f-station-timeline-more').click()
            assert page.locator('#f-station-timeline .station-span').count() == 24
            page.evaluate('''() => {
              const report=stationsUI.reports.file;
              report.timeline=[{interval_id:'first',person_id:'P001',station_id:'s1',state:'at_station',start:0,end:5},{interval_id:'second',person_id:'P001',station_id:'s1',state:'at_station',start:5,end:10}];
              stationsUI.render('file',report);
            }''')
            assert page.locator('#f-station-timeline .station-span').count() == 1
            assert '合并 2 段' in page.locator('#f-station-timeline').inner_text()
            page.locator('#f-station-timeline-mode').select_option('all')
            assert page.locator('#f-station-timeline .station-span').count() == 2
            for width in (390, 768, 1440):
                page.set_viewport_size({'width': width, 'height': 900})
                assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+1'), width
            print('PASS complete evidence remains filterable across desktop/mobile')
            print('PASS pending threshold alerts cite their own interval and distinguish model verification')
            print('PASS long timelines expand and continuous states merge without deleting evidence')
        browser.close()
        assert not errors, errors
        assert not failures, failures


if __name__ == '__main__':
    main()
