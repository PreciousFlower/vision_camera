"""Offline regression for retained workstation alarm lifecycle in the actual UI."""
import copy
import json
from pathlib import Path
import sys

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from webui.station_identity import canonical_window


def main():
    errors, job_requests = [], []
    old_window = {"window_id": "w1", "index": 19, "t_start": 246.9, "t_end": 283.7, "frames": 1,
                  "scene": "1号工位内有人员P004驻留", "people": [
                      {"where": "1号工位", "appearance": "P004", "action": "at_station", "changed": "P004在蓝色区域内"}],
                  "station_review": {"scene": "1号工位内有人员P004驻留", "observations": [
                      {"person_id": "P004", "frame_id": "f_19", "station_id": "a", "state": "at_station", "reason": "P004在蓝色区域内"}], "stations": []},
                  "raw": "原始模型：P004在岗"}
    aliases = {"P004": "P001"}
    mapped_window = canonical_window(old_window, aliases, {"P001"})
    base = {"source": "workstation", "behavior_id": "station_away", "behavior_name": "离岗超时",
            "severity": "medium", "t_start": 10, "t_end": 40, "threshold_time_sec": 20,
            "trigger_time_sec": 20.5, "review_status": "confirmed", "latest_review_status": "confirmed",
            "lifecycle": "active", "reason": "P001 离开区域，已确认离岗", "person_id": "P001",
            "alarm_id": "away-event", "interval_id": "person-interval", "window_ids": ["w1"],
            "active_duration_sec": 30, "verification_delay_sec": 4}
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="msedge", headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.add_init_script("""
          window.SpeechSynthesisUtterance=class {constructor(text){this.text=text;}};
          Object.defineProperty(window,'speechSynthesis',{value:{getVoices:()=>[],addEventListener:()=>{},cancel:()=>{},speak:()=>{}}});
        """)

        def route_request(route):
            url = route.request.url
            if "/static/" in url:
                return route.fulfill(content_type="text/javascript", body=(
                    ROOT / "webui/static" / url.split("?")[0].rsplit("/", 1)[-1]).read_text(encoding="utf-8"))
            if "/frames/" in url:
                return route.fulfill(content_type="image/svg+xml", body='<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10"/>')
            if url.endswith("/api/state"):
                data = {"runtime": {"protocol": 3, "restart_required": False}, "service": {"state": "idle"},
                        "jobs": [], "behaviors": {"enabled": []}, "aggregate": {"mode": "ai"}}
            elif url.endswith("/api/videos"):
                data = {"videos": []}
            elif url.endswith("/api/job/fixture"):
                job_requests.append(url)
                data = {"job_id": "fixture", "window_results": [mapped_window]}
            elif "/api/" in url:
                data = {}
            else:
                return route.fulfill(content_type="text/html", body=(ROOT / "webui/static/index.html").read_text(encoding="utf-8"))
            route.fulfill(content_type="application/json", body=json.dumps(data))

        page.route("**/*", route_request)
        page.goto("http://station-alert.test/", wait_until="networkidle")
        page.wait_for_function("!restoringJobs&&!historyLoading")
        page.evaluate("""monJob='fixture';document.querySelector('#m-feed').replaceChildren(
          windowCard({window_id:'w1',index:1,t_start:10,t_end:40,frames:1,scene:'模型检查',hits:[]},'fixture'))""")

        def render(alert):
            page.evaluate("alert=>renderAlerts([alert],true)", alert)
            assert page.locator("#m-alerts article.alert").count() == 1, "Historical alarm must remain visible"
            assert page.locator("#m-alert-count").inner_text() == "1 / 1 条"

        render(base)
        assert "AI 已确认" in page.locator("#m-alerts").inner_text()
        assert page.locator("#m-feed .act.alerted").count() == 1
        rejected = {**base, "review_status": "rejected", "lifecycle": "withdrawn",
                    "latest_review_status": "rejected", "retracted_time_sec": 42,
                    "retraction_reason": "复核画面证实 P001 仍然在工位", "reason": "复核画面证实 P001 仍然在工位"}
        render(rejected)
        text = page.locator("#m-alerts").inner_text()
        assert "AI 已撤回" in text, text
        assert rejected["retraction_reason"] in text
        assert "AI 已确认" not in text and "大模型已确认" not in text
        assert page.locator("#m-feed .act.alerted").count() == 0, "Withdrawn event cannot paint a window red"
        assert page.locator("#m-feed .alert-badge").count() == 0

        resolved = {**base, "lifecycle": "resolved", "resolution_reason": "returned", "resolved_time_sec": 45}
        render(base)
        render(resolved)
        assert "已恢复" in page.locator("#m-alerts").inner_text()
        for reason in ["observation_lost", "session_ended"]:
            render({**resolved, "resolution_reason": reason})
            text = page.locator("#m-alerts").inner_text()
            assert "已结束" in text, text
            assert "已恢复" not in text, "Loss of visibility or stopping does not prove someone returned"

        uncertain = copy.deepcopy(base)
        uncertain.update(latest_review_status="uncertain", latest_review_time_sec=47,
                         latest_review_reason="人物被遮挡，不能作新的确认")
        render(uncertain)
        text = page.locator("#m-alerts").inner_text()
        assert "最新复核" in text and ("待复核" in text or "不确定" in text), text
        assert uncertain["latest_review_reason"] in text

        # A real Python presentation transformation feeds the real browser
        # /api/job refresh; both old text and the original raw reply are kept.
        page.evaluate("row=>{document.querySelector('#m-feed').replaceChildren(windowCard(row,'fixture'));stationsUI.jobs.monitor={id:'fixture',profile:{mode:'workstation'}};}", old_window)
        assert "P004" in page.locator("#m-feed .scene-desc").inner_text()
        page.evaluate("aliases=>syncStationReport('monitor',{analysis_revision:'revision',time_sec:284,aliases,alerts:[]})", aliases)
        page.wait_for_function("document.querySelector('#m-feed .scene-desc').textContent.includes('P001')")
        assert job_requests, "An alias change must refresh rows through /api/job"
        assert "P004" not in page.locator("#m-feed .scene-desc").inner_text()
        assert "P001" in page.locator("#m-feed .person .where").inner_text()
        assert "P004 → P001" in page.locator("#m-feed").inner_text()
        raw_detail = page.locator("#m-feed details").filter(has=page.locator("summary", has_text="查看模型原始输出"))
        assert "P004" in raw_detail.locator("pre").text_content()
        assert page.locator("#m-feed .scene-desc").count() == 1
        assert not errors, errors
        browser.close()
    print("PASS retained withdrawal/recovery/end/uncertain review; alias API refresh maps P004 to P001 while preserving original raw reply")


if __name__ == "__main__":
    main()
