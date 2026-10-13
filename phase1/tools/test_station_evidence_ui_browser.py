"""Offline actual-page feedback regression for retained station alarm history."""
import json
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]


def main():
    errors = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel='msedge', headless=True)
        page = browser.new_page()
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.add_init_script("""
          window.SpeechSynthesisUtterance=class {constructor(text){this.text=text;}};
          Object.defineProperty(window,'speechSynthesis',{value:{getVoices:()=>[],addEventListener:()=>{},cancel:()=>{},speak:()=>{}}});
        """)

        def route_request(route):
            url = route.request.url
            if '/static/' in url:
                return route.fulfill(content_type='text/javascript', body=(
                    ROOT / 'webui/static' / url.split('?')[0].rsplit('/', 1)[-1]).read_text(encoding='utf-8'))
            if url.endswith('/api/state'):
                data = {'runtime': {'protocol': 3, 'restart_required': False},
                        'service': {'state': 'idle'}, 'jobs': [],
                        'behaviors': {'enabled': []}, 'aggregate': {'mode': 'ai'}}
            elif url.endswith('/api/videos'):
                data = {'videos': []}
            elif '/api/' in url:
                data = {}
            else:
                return route.fulfill(content_type='text/html', body=(
                    ROOT / 'webui/static/index.html').read_text(encoding='utf-8'))
            route.fulfill(content_type='application/json', body=json.dumps(data))

        page.route('**/*', route_request)
        page.goto('http://station-evidence.test/', wait_until='networkidle')

        def show(result):
            page.evaluate("""result=>{
              document.querySelector('#feedback-behavior').replaceChildren(new Option('离岗超时','station_away'));
              evidenceContext={model_results:{station_away:result},feedback:[]};
              loadFeedbackSelection();
            }""", result)
            return page.locator('#feedback-summary').text_content()

        withdrawn = {'origin': 'station_statistics', 'violation': False, 'review_status': 'rejected',
                     'lifecycle': 'withdrawn', 'observable': '旧几何判断', 'retraction_reason': '画面确认人员仍然在岗'}
        text = show(withdrawn)
        assert 'AI 已撤回' in text and withdrawn['retraction_reason'] in text, text
        assert '行为命中' not in text and '旧几何判断' not in text, text
        confirmed = {'origin': 'station_statistics', 'violation': True, 'review_status': 'confirmed',
                     'lifecycle': 'resolved', 'observable': '此前确实离岗', 'resolution_reason': 'returned'}
        assert '已恢复' in show(confirmed)
        for reason in ['observation_lost', 'session_ended', 'interval_closed']:
            text = show({**confirmed, 'resolution_reason': reason})
            assert '已结束' in text and '已恢复' not in text, text
        assert 'AI待复核' in show({**confirmed, 'lifecycle': 'active', 'review_status': 'pending', 'provisional': True})
        assert '复核记录缺失' in show({**confirmed, 'review_status': 'unknown'})
        text = show({'violation': True, 'observable': '明确吸烟'})
        assert '大模型原始判断：行为命中' in text and '明确吸烟' in text, text
        assert not errors, errors
        browser.close()
    print('PASS withdrawn/recovered/ended/pending/unknown station evidence labels; ordinary behavior evidence unchanged')


if __name__ == '__main__':
    main()
