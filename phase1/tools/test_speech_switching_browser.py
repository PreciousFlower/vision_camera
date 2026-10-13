"""Drive actual speech settings without audio, network services, or GPU jobs."""
import json
from pathlib import Path
import unittest

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]


class SpeechSwitchingBrowser(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.playwright = sync_playwright().start()
        cls.browser = cls.playwright.chromium.launch(channel="msedge", headless=True)

    @classmethod
    def tearDownClass(cls):
        cls.browser.close()
        cls.playwright.stop()

    def setUp(self):
        self.errors = []
        self.page = self.browser.new_page()
        self.page.set_default_timeout(1500)
        self.page.on("pageerror", lambda error: self.errors.append(str(error)))
        self.page.add_init_script("""
          window.spoken=[];window.cancelled=0;window.voiceListeners=[];
          window.voiceChanged=()=>voiceListeners.forEach(listener=>listener());
          window.voiceInventory=[
            {name:'Microsoft Huihui - Chinese (Simplified, PRC)',lang:'zh-CN',voiceURI:'cn-female',localService:true},
            {name:'Microsoft Kangkang - Chinese (Simplified, PRC)',lang:'zh-CN',voiceURI:'cn-male',localService:true},
            {name:'Mystery voice',lang:'zh-CN',voiceURI:'cn-unknown',localService:true},
            {name:'Hong Kong Cantonese Natural',lang:'zh-HK',voiceURI:'hk'},
            {name:'Taiwan Mandarin Natural',lang:'zh-TW',voiceURI:'tw'}
          ];
          window.SpeechSynthesisUtterance=class {constructor(text){this.text=text;}};
          Object.defineProperty(window,'speechSynthesis',{value:{
            getVoices:()=>voiceInventory,
            addEventListener:(type,callback)=>{if(type==='voiceschanged')voiceListeners.push(callback);},
            cancel:()=>{cancelled++;const old=spoken.at(-1);if(old?.onerror)old.onerror({error:'canceled'});},
            speak:u=>spoken.push(u)
          }});
        """)

        def handle(route):
            url = route.request.url
            if "/static/" in url:
                return route.fulfill(content_type="text/javascript", body=(
                    ROOT / "webui/static" / url.split("?")[0].rsplit("/", 1)[-1]
                ).read_text(encoding="utf-8"))
            if url.endswith("/api/state"):
                result = {"runtime": {"protocol": 3, "restart_required": False},
                          "service": {"state": "idle"}, "jobs": [],
                          "behaviors": {"enabled": []}, "aggregate": {"mode": "ai"}}
            elif url.endswith("/api/videos"):
                result = {"videos": []}
            elif "/api/" in url:
                result = {}
            else:
                return route.fulfill(content_type="text/html", body=(
                    ROOT / "webui/static/index.html").read_text(encoding="utf-8"))
            route.fulfill(content_type="application/json", body=json.dumps(result))

        self.page.route("**/*", handle)
        self.page.goto("http://speech-switch.test/", wait_until="networkidle")

    def tearDown(self):
        self.page.close()
        self.assertEqual([], self.errors)

    def finish(self):
        self.page.evaluate("spoken.at(-1)?.onend()")

    def test_explicit_mandarin_gender_choices(self):
        values = self.page.locator("#speech-voice option").evaluate_all("options=>options.map(o=>o.value)")
        self.assertIn("mandarin:female", values)
        self.assertIn("mandarin:male", values)
        labels = self.page.locator("#speech-voice option").all_text_contents()
        self.assertTrue(any("女声" in name for name in labels))
        self.assertTrue(any("男声" in name for name in labels))
        self.assertFalse(any(value.endswith(":hk") or value.endswith(":tw") for value in values))
        self.assertEqual("男声 · 康康", self.page.locator("option[value='browser:cn-male']").inner_text())
        self.assertEqual("女声 · 惠惠", self.page.locator("option[value='browser:cn-female']").inner_text())
        self.assertEqual("标准普通话 · Mystery voice", self.page.locator("option[value='browser:cn-unknown']").inner_text())
        self.page.locator("#speech-voice").select_option("mandarin:male")
        self.assertEqual("cn-male", self.page.evaluate("spoken.at(-1)?.voice?.voiceURI"))
        self.assertEqual("mandarin:male", self.page.evaluate("speech.voice"))

    def test_voice_switch_restarts_current_preview_immediately(self):
        self.page.locator("#speech-test").click()
        self.assertEqual(1, self.page.evaluate("spoken.length"))
        self.page.locator("#speech-voice").select_option("browser:cn-male")
        self.assertEqual(2, self.page.evaluate("spoken.length"), "Changing voice must immediately restart playback")
        self.assertEqual("cn-male", self.page.evaluate("spoken.at(-1).voice.voiceURI"))
        self.assertEqual(self.page.evaluate("spoken[0].text"), self.page.evaluate("spoken[1].text"))
        self.assertTrue(self.page.evaluate("speech.enabled"), "The canceled old voice cannot disable the new playback")

    def test_voice_switch_previews_when_idle_and_persists(self):
        self.page.locator("#speech-voice").select_option("browser:cn-male")
        self.assertEqual(1, self.page.evaluate("spoken.length"), "An idle voice change should preview the chosen voice")
        self.assertEqual("cn-male", self.page.evaluate("spoken.at(-1).voice.voiceURI"))
        self.finish()
        self.page.reload(wait_until="networkidle")
        self.assertEqual("browser:cn-male", self.page.locator("#speech-voice").input_value())
        self.assertEqual(0, self.page.evaluate("spoken.length"), "Reload cannot play audio without a fresh user gesture")

    def test_rate_selection_restarts_same_text_at_distinct_actual_rates(self):
        rates = self.page.locator("#speech-rate option").evaluate_all("options=>options.map(o=>o.value)")
        self.assertEqual(["0.75", "1", "1.3", "1.5"], rates)
        self.page.locator("#speech-test").click()
        text = self.page.evaluate("spoken.at(-1).text")
        for rate in ["0.75", "1.3", "1.5"]:
            previous = self.page.evaluate("spoken.length")
            self.page.locator("#speech-rate").select_option(rate)
            self.assertEqual(previous + 1, self.page.evaluate("spoken.length"))
            self.assertEqual(float(rate), self.page.evaluate("spoken.at(-1).rate"))
            self.assertEqual(text, self.page.evaluate("spoken.at(-1).text"))
        self.finish()
        self.page.reload(wait_until="networkidle")
        self.assertEqual("1.5", self.page.locator("#speech-rate").input_value())

    def test_setting_change_restarts_alert_and_preserves_queue(self):
        self.page.evaluate("""speech.enable(true);speech.notify('real-alert',[
          {behavior_id:'smoking',behavior_name:'吸烟',t_start:1},
          {behavior_id:'littering',behavior_name:'乱扔垃圾',t_start:2}
        ])""")
        alert = self.page.evaluate("speech.active.text")
        self.page.locator("#speech-voice").select_option("browser:cn-male")
        self.assertEqual(2, self.page.evaluate("spoken.length"))
        self.assertEqual(alert, self.page.evaluate("spoken.at(-1).text"))
        self.assertEqual("real-alert", self.page.evaluate("speech.active.job"))
        self.assertEqual(1, self.page.evaluate("speech.queue.length"))
        self.finish()
        self.assertIn("乱扔垃圾", self.page.evaluate("spoken.at(-1).text"))
        self.finish()
        self.assertIsNone(self.page.evaluate("speech.active"))

    def test_voice_list_refresh_uses_current_objects_and_keeps_gender_setting(self):
        values = self.page.locator("#speech-voice option").evaluate_all("options=>options.map(o=>o.value)")
        self.assertIn("mandarin:male", values)
        self.page.locator("#speech-voice").select_option("mandarin:male")
        self.finish()
        self.page.evaluate("voiceInventory=voiceInventory.map(voice=>({...voice,version:2}));voiceChanged()")
        self.assertEqual("mandarin:male", self.page.locator("#speech-voice").input_value())
        self.page.locator("#speech-test").click()
        self.assertEqual(2, self.page.evaluate("spoken.at(-1).voice.version"))
        self.assertEqual("cn-male", self.page.evaluate("spoken.at(-1).voice.voiceURI"))

    def test_empty_voice_inventory_keeps_male_then_resumes_when_loaded(self):
        self.page.evaluate("speech.enable(true);window.savedVoices=voiceInventory;voiceInventory=[];voiceChanged()")
        self.page.locator("#speech-voice").select_option("mandarin:male")
        self.assertEqual("mandarin:male", self.page.locator("#speech-voice").input_value())
        self.assertEqual("mandarin:male", self.page.evaluate("speech.voice"))
        self.assertTrue(self.page.evaluate("speech.enabled"))
        self.assertIn("男声", self.page.locator("#speech-status").inner_text())
        self.assertNotIn("检查声音权限", self.page.locator("#speech-status").inner_text())
        self.page.evaluate("voiceInventory=savedVoices;voiceChanged()")
        self.assertEqual("cn-male", self.page.evaluate("spoken.at(-1).voice.voiceURI"))
        self.finish()

    def test_legacy_rate_is_visible_and_saved_without_silent_reset(self):
        for rate in [.9, 1.1]:
            with self.subTest(rate=rate):
                self.page.evaluate("rate=>localStorage.setItem('civility-speech-settings',JSON.stringify({voice:'mandarin:male',rate,volume:.7}))", rate)
                self.page.reload(wait_until="networkidle")
                self.assertEqual(str(rate), self.page.locator("#speech-rate").input_value())
                self.assertIn(f"{rate:.2f}×", self.page.locator("#speech-rate option:checked").inner_text())
                self.assertEqual("mandarin:male", self.page.locator("#speech-voice").input_value())
                self.page.locator("#speech-test").click()
                self.assertEqual(rate, self.page.evaluate("spoken.at(-1).rate"))
                self.assertEqual(.7, self.page.evaluate("spoken.at(-1).volume"))
                self.assertEqual(rate, self.page.evaluate("JSON.parse(localStorage.getItem('civility-speech-settings')).rate"))
                self.finish()

    def test_voice_unavailable_error_keeps_alert_and_specific_notice(self):
        self.page.locator("#speech-voice").select_option("mandarin:male")
        self.finish()
        self.page.evaluate("speech.enable(true);speech.notify('recoverable',[{behavior_id:'smoking',behavior_name:'吸烟',t_start:1}]);spoken.at(-1).onerror({error:'voice-unavailable'})")
        self.assertIn("普通话声音暂不可用", self.page.locator("#speech-status").inner_text())
        self.assertNotIn("检查声音权限", self.page.locator("#speech-status").inner_text())
        self.assertTrue(self.page.evaluate("speech.enabled"))
        self.assertEqual("recoverable", self.page.evaluate("speech.queue[0].job"))
        self.page.evaluate("voiceChanged()")
        self.assertEqual("recoverable", self.page.evaluate("speech.active.job"))
        self.assertEqual("cn-male", self.page.evaluate("spoken.at(-1).voice.voiceURI"))
        self.finish()

    def test_initial_setting_selection_activates_checked_alarm_speech(self):
        self.assertTrue(self.page.locator("#speech-enabled").is_checked())
        self.assertFalse(self.page.evaluate("speech.enabled"))
        self.page.locator("#speech-voice").select_option("mandarin:male")
        self.assertTrue(self.page.evaluate("speech.enabled"), "A setting change is a user gesture and must activate a checked alarm switch")
        self.finish()
        self.page.evaluate("speech.notify('initial-selection',[{behavior_id:'smoking',behavior_name:'吸烟',t_start:1}])")
        self.assertEqual("initial-selection", self.page.evaluate("speech.active.job"))
        self.assertEqual("cn-male", self.page.evaluate("spoken.at(-1).voice.voiceURI"))
        self.finish()

    def test_hard_audio_error_synchronizes_switch_and_retains_retryable_alert(self):
        self.page.locator("#speech-test").click()
        self.finish()
        self.page.evaluate("speech.notify('hardware-failure',[{behavior_id:'smoking',behavior_name:'吸烟',t_start:1}]);spoken.at(-1).onerror({error:'audio-hardware'})")
        self.assertFalse(self.page.evaluate("speech.enabled"))
        self.assertFalse(self.page.locator("#speech-enabled").is_checked(), "A hard audio error must clear the switch as well as engine enablement")
        self.assertEqual("false", self.page.evaluate("localStorage.getItem('civility-speech-enabled')"))
        self.assertEqual("hardware-failure", self.page.evaluate("speech.queue[0].job"))
        self.page.locator("#speech-voice").select_option("mandarin:male")
        self.assertFalse(self.page.evaluate("speech.enabled"), "A setting change cannot silently turn a disabled alarm back on")
        self.assertFalse(self.page.locator("#speech-enabled").is_checked())
        self.finish()
        self.assertIsNone(self.page.evaluate("speech.active"))
        self.page.locator("#speech-enabled").check()
        self.assertTrue(self.page.evaluate("speech.enabled"))
        self.assertEqual("true", self.page.evaluate("localStorage.getItem('civility-speech-enabled')"))
        self.finish()
        self.assertEqual("hardware-failure", self.page.evaluate("speech.active.job"))
        self.assertEqual("cn-male", self.page.evaluate("spoken.at(-1).voice.voiceURI"))
        self.finish()


if __name__ == "__main__":
    unittest.main(verbosity=2)
