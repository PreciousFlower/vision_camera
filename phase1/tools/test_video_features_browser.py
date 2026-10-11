"""Browser integration without making the user's speakers play aloud."""
import json
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "runs/features-20261011"


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge", headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1000}, reduced_motion="reduce")
        page.add_init_script("""window.spoken=[];
          window.SpeechSynthesisUtterance=class {constructor(text){this.text=text;}};
          Object.defineProperty(window,'speechSynthesis',{value:{getVoices:()=>[{lang:'zh-CN'}],
            cancel:()=>{},speak:u=>{window.spoken.push(u.text);setTimeout(()=>u.onend&&u.onend(),5);}}});
          window.EventSource=class {static CLOSED=2;constructor(){this.readyState=1;window.stream=this;}close(){}};
        """)
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        def respond(route):
            url = route.request.url
            if "/static/alert_speech.js" in url:
                return route.fulfill(content_type="text/javascript", body=(ROOT / "webui/static/alert_speech.js").read_text(encoding="utf-8"))
            if "/frames/" in url:
                return route.fulfill(content_type="image/svg+xml", body='<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="720"><rect width="1280" height="720" fill="#237969"/></svg>')
            if "/api/state" in url:
                data = {"runtime": {"protocol": 3, "restart_required": False}, "service": {"state": "ready", "model": "deployed"}, "jobs": [], "aggregate": {}, "behaviors": {"enabled": []}}
            elif url.endswith("/api/videos"):
                data = {"videos": [{"video_id": "fixture", "filename": "吸烟.mp4", "windows_seen": 3, "source_available": False}]}
            elif url.endswith("/ask"):
                data = {"answer": "<script>不可执行</script> 在 2 秒的引用画面中可见手持烟支。", "model": "Qwen3-VL", "sampled_frames": 12,
                        "limitations": "仅抽样画面", "citations": [{"image_url": "/api/videos/fixture/frames/f_6", "time_sec": 2, "sample_number": 7, "description": "手持烟支"}]}
            elif "/api/" in url:
                data = {}
            else:
                return route.fulfill(content_type="text/html", body=(ROOT / "webui/static/index.html").read_text(encoding="utf-8"))
            route.fulfill(content_type="application/json", body=json.dumps(data))
        page.route("**/*", respond)
        page.goto("http://features.test/", wait_until="networkidle")
        page.wait_for_function("!runtimeOutdated&&!restoringJobs")
        page.locator("#speech-test").click()
        assert len(page.evaluate("window.spoken")) == 1
        page.evaluate("""() => {
          speech.enable(true); speech.begin('new');newSpeechRuns.add('new');streamJob('new');
          const alerts=[{behavior_id:'smoking',behavior_name:'吸烟',severity:'medium',t_start:2,reason:'可见烟支'}];
          stream.onmessage({data:JSON.stringify({type:'hello',summary:{state:'running',alerts:[]}})});
          stream.onmessage({data:JSON.stringify({type:'alerts',alerts})});
          stream.onmessage({data:JSON.stringify({type:'alerts',alerts})});
        }""")
        page.wait_for_function("window.spoken.length===2")
        assert "吸烟" in page.evaluate("window.spoken[1]")
        page.evaluate("""() => {stream.onmessage({data:JSON.stringify({type:'hello',summary:{state:'running',alerts:[{behavior_id:'smoking',severity:'medium',t_start:2}]}})});} """)
        assert len(page.evaluate("window.spoken")) == 2
        page.locator('aside button[data-tab="qa"]').click()
        page.wait_for_function("document.querySelector('#qa-video').value==='fixture'")
        page.locator("#qa-question").fill("哪一帧有人吸烟？")
        page.locator("#qa-ask").click()
        page.locator("#qa-answers .card").wait_for()
        assert page.locator("#qa-answers script").count() == 0
        assert "2.00 秒" in page.locator("#qa-answers").text_content()
        page.locator("#qa-answers img").click()
        assert page.locator("#image-viewer").is_visible()
        assert page.locator("#image-viewer img").evaluate("el=>getComputedStyle(el).objectFit") == "contain"
        page.keyboard.press("Escape")
        page.screenshot(path=str(OUTPUT / "video-qa-desktop.png"), full_page=True)
        page.set_viewport_size({"width": 390, "height": 844})
        assert page.evaluate("document.documentElement.scrollWidth<=innerWidth+1")
        page.screenshot(path=str(OUTPUT / "video-qa-mobile.png"), full_page=True)
        assert not errors, errors
        result = {"speaker_test": "mocked, no audible playback", "new_alert_dedup": True, "reconnect_no_replay": True,
                  "qa_answer_and_citations": True, "escaped_model_output": True, "full_frame_viewer": True, "mobile_no_overflow": True}
        (OUTPUT / "browser.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        browser.close()
        print("Browser feature integration passed", result)


if __name__ == "__main__":
    main()
