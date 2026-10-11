"""Stability reports must accept the same results returned by real backends."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from civility.aggregate import stability_report
from civility.config import Behavior, BehaviorSet
from civility.prompts import sanitize


class StabilityReportTests(unittest.TestCase):
    def setUp(self):
        self.bs = BehaviorSet([Behavior('fixture', '测试行为', '可观察行为')], {})

    def raw(self, violation):
        return {'violations': [{'behavior_id': 'fixture', 'violation': violation,
                                'confidence': .9, 'observable': '测试画面可见行为'}]}

    def test_real_sanitized_results_detect_disagreement(self):
        results = [sanitize(self.raw(v), self.bs) for v in (True, False)]
        report = stability_report({'w': results}, self.bs)
        self.assertEqual(report['fully_stable'], 0)
        self.assertEqual(report['overall_agreement'], 0)
        self.assertEqual(report['positive_rate_per_behavior']['fixture'], .5)

    def test_legacy_raw_and_normalized_formats_have_equal_reports(self):
        raw = {'a': [self.raw(True), self.raw(True)],
               'b': [self.raw(False), self.raw(True)]}
        normalized = {wid: [sanitize(obj, self.bs) for obj in items]
                      for wid, items in raw.items()}
        expected = stability_report(raw, self.bs)
        self.assertEqual(expected['overall_agreement'], .5)
        self.assertEqual(expected['positive_rate_per_behavior']['fixture'], .75)
        self.assertEqual(stability_report(normalized, self.bs), expected)

    def test_mixed_formats_and_absent_items_keep_existing_semantics(self):
        repeated = {'a': [self.raw(True), {'violations': {}}],
                    'b': [self.raw(False), sanitize(self.raw(False), self.bs)],
                    'single': [self.raw(True)]}
        report = stability_report(repeated, self.bs)
        self.assertEqual(report['windows_checked'], 2)
        self.assertEqual(report['fully_stable'], 1)
        self.assertEqual(report['positive_rate_per_behavior']['fixture'], .3333)


if __name__ == '__main__':
    unittest.main()
