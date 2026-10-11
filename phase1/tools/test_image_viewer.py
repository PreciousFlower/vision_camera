"""Offline browser regression for full-frame image viewing; never starts a backend.

Requires Playwright for Python and installed Microsoft Edge. Run from phase1:
    python tools/test_image_viewer.py
All requests are fulfilled locally, including the real page and fixture frames.
"""
from __future__ import annotations

import json
from pathlib import Path

from playwright.sync_api import sync_playwright


ROOT = Path(__file__).resolve().parents[1]
HTML = ROOT / "webui" / "static" / "index.html"
OUTPUT = ROOT / "runs" / "image-viewer-validation"


def frame(width: int, height: int) -> str:
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
      <rect width="100%" height="100%" fill="#173547"/>
      <rect x="6" y="6" width="{width - 12}" height="{height - 12}" fill="none" stroke="#55e6be" stroke-width="12"/>
      <path d="M0 0 L{width} {height} M{width} 0 L0 {height}" stroke="#7390a4" stroke-width="3"/>
      <circle cx="{width / 2}" cy="{height / 2}" r="{min(width, height) / 5}" fill="#07877e"/>
      <text x="50%" y="50%" dominant-baseline="middle" text-anchor="middle" fill="white" font-size="36">{width} x {height}</text>
      <text x="24" y="52" fill="white" font-size="24">TOP LEFT</text>
      <text x="{width - 24}" y="{height - 26}" text-anchor="end" fill="white" font-size="24">BOTTOM RIGHT</text>
    </svg>'''


def assert_full_frame(page) -> dict:
    # The legacy implementation enlarges the thumbnail itself; the fixed page
    # uses a dedicated dialog. Both paths are measured against the same symptom.
    image = page.locator("#image-viewer[open] img, img.big").first
    image.wait_for(state="visible")
    result = image.evaluate("""img => {
      const r=img.getBoundingClientRect(), s=getComputedStyle(img);
      return {width:r.width,height:r.height,x:r.x,y:r.y,
        viewportWidth:innerWidth,viewportHeight:innerHeight,
        naturalRatio:img.naturalWidth/img.naturalHeight,objectFit:s.objectFit};
    }""")
    ratio = result["width"] / result["height"]
    fits = result["objectFit"] == "contain" or abs(ratio - result["naturalRatio"]) < 0.01
    assert fits, f"Image crops or distorts the original frame: {json.dumps(result)}"
    assert result["x"] >= -1 and result["y"] >= -1, result
    assert result["x"] + result["width"] <= result["viewportWidth"] + 1, result
    assert result["y"] + result["height"] <= result["viewportHeight"] + 1, result
    return result


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge", headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1000}, reduced_motion="reduce")
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        fixture = {"width": 1280, "height": 720}

        def respond(route):
            url = route.request.url
            if '/static/alert_speech.js' in url:
                route.fulfill(status=200, content_type='text/javascript', body=(ROOT / 'webui/static/alert_speech.js').read_text(encoding='utf-8'))
            elif "/frames/" in url or "/fixture-frame" in url:
                route.fulfill(status=200, content_type="image/svg+xml", body=frame(**fixture))
            elif "/api/" in url:
                route.fulfill(status=200, content_type="application/json", body="{}")
            else:
                route.fulfill(status=200, content_type="text/html", body=HTML.read_text(encoding="utf-8"))

        page.route("**/*", respond)
        for viewport in ({"width": 1440, "height": 1000}, {"width": 390, "height": 844}):
            page.set_viewport_size(viewport)
            for width, height in ((1280, 720), (540, 960), (720, 720), (2048, 256)):
                fixture.update(width=width, height=height)
                page.goto("http://image-viewer.test/", wait_until="networkidle")
                page.locator('aside button[data-tab="detect"]').click()
                page.evaluate("""() => {
                  document.querySelector('#windows').replaceChildren(windowCard({
                    window_id:'fixture',index:1,t_start:0,t_end:2,frames:4,frame0:0,
                    scene:'测试画面：四周绿色边框应完整显示，文字保持原样。',
                    people:[{where:'中央', action:'手持物品，动作待确认'}],hits:[],risks:[]
                  }, 'offline'));
                }""")
                thumb = page.locator("#windows .thumb")
                thumb.scroll_into_view_if_needed()
                page.wait_for_function("document.querySelector('#windows .thumb').naturalWidth > 0")
                before = thumb.bounding_box()
                thumb.click()
                try:
                    result = assert_full_frame(page)
                except AssertionError:
                    page.screenshot(path=str(OUTPUT / "before-cropped.png"))
                    raise
                after = thumb.bounding_box()
                assert before == after, f"Opening image changes the card layout: {before} -> {after}"
                page.screenshot(path=str(OUTPUT / f"viewer-{viewport['width']}-{width}x{height}.png"))
                page.keyboard.press("Escape")
                assert page.locator("#image-viewer[open]").count() == 0
                assert thumb.evaluate("img => document.activeElement === img"), "Focus must return to the clicked thumbnail"
                thumb.press("Enter")
                assert_full_frame(page)
                page.locator("#image-viewer-close").click()
                print(f"PASS {viewport['width']}px viewport / {width}x{height} frame / fit={result['objectFit']}")
        # Exercise the same shared card under the monitor's scrolling feed and
        # the two pre-existing live/evidence image controls.
        page.locator('aside button[data-tab="monitor"]').click()
        page.evaluate("""() => {
          const d={window_id:'review',index:1,t_start:2,t_end:4,frames:4,frame0:4,
            scene:'模型称手部有烟雾，需核对原始画面。',
            people:[{where:'中央',action:'模型称人物吸烟'}],hits:[],risks:[],
            evidence_warnings:[{behavior_id:'smoking',name:'吸烟',
              issues:['缺少可辨认的香烟证据','<img src=x onerror=alert(1)>']}]};
          document.querySelector('#windows').replaceChildren(windowCard(d,'offline'));
          document.querySelector('#m-feed').replaceChildren();
          addAction(d,'offline');
          for(const id of ['m-live','m-evidence']){
            const img=document.getElementById(id);
            img.src='/fixture-frame';img.style.display='block';
          }
        }""")
        for container in ("#windows", "#m-feed"):
            text = page.locator(container).text_content()
            assert "待复核：证据不足，未告警" in text
            assert "以下为 AI 原始描述，待复核" in text
            assert "<img src=x onerror=alert(1)>" in text
            assert page.locator(f"{container} .evidence-review img").count() == 0
        for selector in ("#m-feed .thumb", "#m-evidence", "#m-live"):
            source = page.locator(selector)
            source.scroll_into_view_if_needed()
            source.press("Space")
            assert_full_frame(page)
            page.set_viewport_size({"width": 844, "height": 390})
            assert_full_frame(page)
            page.locator("#image-viewer-image").click()
            assert source.evaluate("img => document.activeElement === img")
            page.set_viewport_size({"width": 390, "height": 844})
            print(f"PASS shared viewer {selector} / keyboard / resize / focus")
        page.locator("#m-feed").scroll_into_view_if_needed()
        page.screenshot(path=str(OUTPUT / "evidence-review-mobile.png"))
        print("PASS pending evidence warnings in file and monitor cards / escaped text")
        page.locator('aside button[data-tab="detect"]').click()
        page.locator('#windows details summary').click()
        assert page.locator('#windows .sequence-frame').count() == 4
        source=page.locator('#windows .sequence-frame').nth(3)
        source.scroll_into_view_if_needed()
        source.press('Enter')
        assert_full_frame(page)
        assert '/3?' in page.locator('#image-viewer-image').get_attribute('src')
        page.keyboard.press('Escape')
        print('PASS complete window sequence review / exact frame / full image')
        assert not errors, errors
        browser.close()


if __name__ == "__main__":
    main()
