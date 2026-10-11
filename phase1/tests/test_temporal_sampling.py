from pathlib import Path
import sys
import copy
import tempfile
from unittest.mock import patch
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from civility.extract import build_windows
from civility.extract import extract_frames


class TemporalSamplingTests(unittest.TestCase):
    def test_frame_limit_covers_end_of_window_instead_of_only_start(self):
        frames = [(i / 2, f"f_{i:05d}.jpg") for i in range(12)]
        windows = build_windows("clip.mp4", frames, window_stride_sec=6, max_frames_per_window=4)
        self.assertIn("t=5.5s", windows[0].labels)
        self.assertEqual(len(windows[0].frames), 4)

    def test_overlapping_window_contains_hold_release_landing_and_walk_away(self):
        frames = [(i / 2, f"f_{i:05d}.jpg") for i in range(20)]
        windows = build_windows("clip.mp4", frames, window_stride_sec=2,
                                window_duration_sec=6, max_frames_per_window=12)
        self.assertTrue(any(set(["t=6.5s", "t=7.0s", "t=7.5s", "t=8.0s"])
                            <= set(w.labels) for w in windows))

    def test_does_not_repeat_short_tail_after_complete_context(self):
        frames = [(i / 2, f"f_{i:05d}.jpg") for i in range(20)]
        windows = build_windows("clip.mp4", frames, window_stride_sec=2,
                                window_duration_sec=6, max_frames_per_window=12)
        self.assertEqual([w.t_start for w in windows], [0, 2, 4])
        self.assertEqual(len(windows[-1].frames), 12)

    def test_sparse_sample_frame_endpoint_returns_actual_selected_frame(self):
        from fastapi.testclient import TestClient
        from webui import server as S
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frames = root / "job"
            frames.mkdir()
            for i in (0, 4, 9, 11):
                (frames / f"f_{i:05d}.jpg").write_bytes(f"frame-{i}".encode())
            cfg = copy.deepcopy(S.STATE.cfg)
            cfg["paths"]["frames_dir"] = directory
            job = S.JobRecord("job", "clip.mp4")
            job.window_results = [{"window_id": "w", "frames": 4, "frame0": 0,
                                   "frame_indices": [0, 4, 9, 11]}]
            with patch.object(S.STATE, "cfg", cfg), patch.object(S.STATE, "jobs", {"job": job}):
                with TestClient(S.create_app()) as client:
                    response = client.get("/api/job/job/frames/w/2")
                    self.assertEqual(response.status_code, 200)
                    self.assertEqual(response.content, b"frame-9")

    def test_sampling_cache_changes_when_fps_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "clip.mp4"
            video.write_bytes(b"fixture")
            output = root / "frames"
            def extract(*args, **kwargs):
                (output / "f_00000.jpg").write_bytes(b"frame")
            with patch("civility.extract.find_ffmpeg", return_value=("ffmpeg", "ffprobe")), \
                    patch("civility.extract.subprocess.run", side_effect=extract) as run:
                extract_frames(video, output, fps=1)
                extract_frames(video, output, fps=1)
                self.assertEqual(run.call_count, 1)
                extract_frames(video, output, fps=2)
                self.assertEqual(run.call_count, 2)

    def test_live_context_survives_batch_boundaries_and_stays_bounded(self):
        from webui import server as S
        cfg = copy.deepcopy(S.STATE.cfg)
        cfg["live"].update(continuous_batch_windows=1, context_sec=4,
                           window_stride_sec=2, frames_per_window=4)
        cfg["extract"]["max_frames_per_window"] = 12
        batches = [[[(i / 2, f"f_{i:05d}.jpg") for i in range(start, start + 4)]]
                   for start in (0, 4, 8, 12)]
        job = S.JobRecord("live-context", "clip.mp4", continuous=True)
        with patch.object(S.STATE, "cfg", cfg), \
                patch("webui.capture.capture_windows", side_effect=batches):
            windows = [S.capture_live_windows(job, "clip.mp4")[0] for _ in batches]
        self.assertEqual(windows[2].frame_indices, list(range(12)))
        self.assertEqual(windows[3].frame_indices, list(range(4, 16)))
        self.assertEqual(len({w.window_id for w in windows}), 4)
        self.assertLessEqual(len(job.context_frames), 12)

    def test_rewound_live_source_does_not_reuse_old_action_frames(self):
        from webui import server as S
        job = S.JobRecord("rewound", "clip.mp4", continuous=True,
                          context_frames=[(8, "f_00008.jpg"), (9, "f_00009.jpg")])
        with patch("webui.capture.capture_windows", return_value=[
                [(0, "f_00100.jpg"), (1, "f_00101.jpg")]]):
            window = S.capture_live_windows(job, "clip.mp4")[0]
        self.assertEqual(window.frame_indices, [100, 101])


if __name__ == "__main__":
    unittest.main()
