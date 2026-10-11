"""Motion gate regressions for context frames and unreadable input."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from webui.capture import frame_motion
from webui import server as S


class MotionGateTests(unittest.TestCase):
    def setUp(self):
        self.still = np.zeros((24, 32), dtype=np.uint8)
        self.moving = np.full((24, 32), 120, dtype=np.uint8)

    def measure(self, images):
        paths = [f'frame-{i}.jpg' for i in range(len(images))]
        with patch('cv2.imread', side_effect=images) as read:
            result = frame_motion(paths)
        return result, read.call_count

    def test_new_movement_after_static_context_is_measured(self):
        score, reads = self.measure([self.still] * 8 + [self.moving] * 4)
        self.assertEqual(score, 120)
        self.assertEqual(reads, 12)

    def test_complete_static_window_remains_still(self):
        score, reads = self.measure([self.still] * 12)
        self.assertEqual(score, 0)
        self.assertEqual(reads, 12)

    def test_fewer_than_two_readable_images_is_unknown(self):
        for images in ([], [None] * 4, [self.still], [None, self.still, None]):
            with self.subTest(count=len(images)):
                score, _ = self.measure(images)
                self.assertLess(score, 0)
                job = S.JobRecord('unknown', 'fixture', continuous=True)
                job.window_results = [{'violations': {}, 'risks': []}]
                self.assertFalse(S._should_skip_window(job, score, .5, 2, 30, 0))

    def test_failed_read_cannot_hide_later_movement(self):
        images = [None] * 4 + [self.still, self.moving]
        score, reads = self.measure(images)
        self.assertEqual(score, 120)
        self.assertEqual(reads, len(images))


if __name__ == '__main__':
    unittest.main()
