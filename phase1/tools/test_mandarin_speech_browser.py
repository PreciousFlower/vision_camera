"""Offline UI regression for Mainland Mandarin selection and preview interruption."""
import json
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]


def main():
    errors, synthesis_requests = [], []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="msedge", headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.add_init_script("""
          localStorage.setItem('civility-speech-settings',JSON.stringify({voice:'neural:xiaoxiao',rate:1,volume:.6}));
          window.spoken=[];window.cancelled=0;
          window.voiceInventory=[
            {name:'Hong Kong Cantonese Natural',lang:'zh-HK',voiceURI:'hk'},
            {name:'Taiwan Mandarin Natural',lang:'zh-TW',voiceURI:'tw'},
            {name:'Cantonese',lang:'yue-CN',voiceURI:'yue'},
            {name:'Microsoft Huihui',lang:'zh-CN',voiceURI:'cn',localService:true},
            {name:'Microsoft Kangkang',lang:'zh-CN',voiceURI:'cn-male',localService:true}
          ];
          window.SpeechSynthesisUtterance=class {constructor(text){this.text=text;}};
          Object.defineProperty(window,'speechSynthesis',{value:{
            getVoices:()=>voiceInventory,addEventListener:()=>{},
            cancel:()=>cancelled++,speak:u=>spoken.push(u)
          }});
        """)

        def route_request(route):
            url = route.request.url
            if "/static/" in url:
                return route.fulfill(content_type="text/javascript", body=(
                    ROOT / "webui/static" / url.split("?")[0].rsplit("/", 1)[-1]
                ).read_text(encoding="utf-8"))
            if "/api/speech/" in url:
                synthesis_requests.append(url)
                return route.fulfill(status=503, body="{}")
            if url.endswith("/api/state"):
                data = {"runtime": {"protocol": 3, "restart_required": False},
                        "service": {"state": "idle"}, "jobs": [],
                        "behaviors": {"enabled": []}, "aggregate": {"mode": "ai"}}
            elif url.endswith("/api/videos"):
                data = {"videos": []}
            elif "/api/" in url:
                data = {}
            else:
                return route.fulfill(content_type="text/html", body=(
                    ROOT / "webui/static/index.html").read_text(encoding="utf-8"))
            route.fulfill(content_type="application/json", body=json.dumps(data))

        page.route("**/*", route_request)
        page.goto("http://speech.test/", wait_until="networkidle")
        assert page.locator("#speech-voice").input_value() == "auto"
        assert page.locator("#speech-voice option").evaluate_all(
            "options=>options.map(option=>option.value)") == ["auto", "mandarin:female", "mandarin:male", "browser:cn", "browser:cn-male"]
        page.locator("#speech-test").click()
        assert page.evaluate("spoken.length") == 1
        assert page.evaluate("spoken[0].voice.voiceURI") == "cn"
        assert page.evaluate("spoken[0].volume") == .6
        page.locator("#speech-test").click()
        assert page.evaluate("spoken.length") == 2
        assert page.evaluate("cancelled") >= 2
        page.evaluate("spoken[0].onend()")
        assert page.evaluate("!!speech.active&&speech.active.preview")
        page.evaluate("spoken[1].onend()")
        assert "标准普通话" in page.locator("#speech-status").inner_text()

        page.locator("#speech-voice").select_option("browser:cn-male")
        page.locator("#speech-test").click()
        assert page.evaluate("spoken.at(-1).voice.voiceURI") == "cn-male"
        page.evaluate("spoken.at(-1).onend()")
        page.evaluate("""speech.notify('real-alerts',[
          {behavior_id:'smoking',behavior_name:'吸烟',t_start:1},
          {behavior_id:'littering',behavior_name:'乱扔垃圾',t_start:2}
        ])""")
        page.locator("#speech-test").click()
        assert page.evaluate("speech.queue.filter(item=>!item.preview).length") == 2
        page.evaluate("spoken.at(-1).onend()")
        assert page.evaluate("speech.active.job") == "real-alerts"
        page.evaluate("spoken.at(-1).onend();spoken.at(-1).onend()")
        assert page.evaluate("speech.active") is None

        page.evaluate("voiceInventory=voiceInventory.filter(voice=>voice.lang!=='zh-CN');refreshSpeechVoices()")
        page.locator("#speech-test").click()
        assert page.locator("#speech-voice").input_value() == "browser:cn-male"
        assert "所选标准普通话声音暂不可用" in page.locator("#speech-status").inner_text()
        assert page.evaluate("speech.enabled"), "Loading voices cannot disable real alarm playback"
        assert not synthesis_requests, synthesis_requests
        assert not errors, errors
        browser.close()
    print("PASS Mandarin UI migration/filter, repeated preview restart, alert resumption, missing voice notice and zero synthesis requests")


if __name__ == "__main__":
    main()
