"""Archive browsing/deletion boundaries and natural speech, without a GPU."""
import asyncio
import json
from pathlib import Path
import sys
import tempfile
import threading
import types
import unittest
from unittest.mock import AsyncMock, Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fastapi.testclient import TestClient
from webui import server as S
from webui.speech_service import SpeechService, validate_speech
from webui.video_library import VideoLibrary


class WorkspaceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.library = VideoLibrary(self.root / 'archives', self.root / 'frames', self.root / 'uploads')
        self.saved = S.STATE.jobs, S.STATE.uploads
        S.STATE.jobs, S.STATE.uploads = {}, {}
        self.library_patch = patch.object(S, 'video_library', return_value=self.library)
        self.runtime_patch = patch.object(S, 'ensure_current_runtime')
        self.library_patch.start(); self.runtime_patch.start()
        self.client = TestClient(S.create_app())
        self.job = self.archive('one')
        self.other = self.archive('two')

    def archive(self, video_id):
        job = S.JobRecord(video_id, video_id + '.mp4', state='done', windows_seen=1)
        self.library.uploads_root.mkdir(parents=True, exist_ok=True)
        job._archive_source = self.library.uploads_root / (video_id + '.mp4')
        job._archive_source.write_bytes(b'original video')
        row = {'window_id': 'w0', 't_start': 0, 't_end': 4, 'frames': 2,
               'frame_indices': [0, 3], 'frame_labels': ['t=0.0s', 't=1.0s'],
               'scene': 'visible action', 'source_path': 'must not expose',
               'violations': {'smoking': {'violation': True, 'confidence': .9, 'observable': 'visible cigarette'}}}
        self.library.begin(job); self.library.record(job, row); self.library.save(job)
        directory = self.library.frames_root / video_id
        directory.mkdir(parents=True)
        (directory / 'f_00000.jpg').write_bytes(b'evidence')
        S.STATE.jobs[video_id] = job
        S.STATE.uploads[video_id] = job._archive_source
        return job

    def tearDown(self):
        self.client.close(); self.library_patch.stop(); self.runtime_patch.stop()
        S.STATE.jobs, S.STATE.uploads = self.saved
        self.temp.cleanup()

    def test_history_details_survive_restart_and_do_not_expose_paths(self):
        S.STATE.jobs.clear()
        response = self.client.get('/api/videos/one/analysis')
        self.assertEqual(response.status_code, 200)
        self.assertNotIn('source_path', response.text)
        result = response.json()
        self.assertEqual(result['window_results'][0]['hits'][0]['behavior_id'], 'smoking')
        self.assertEqual(result['window_results'][0]['archive_revision'], result['analysis_revision'])

    def test_delete_removes_only_owned_video_and_cannot_resurrect(self):
        self.assertEqual(self.client.delete('/api/videos/one').status_code, 200)
        self.assertFalse(self.library.folder('one').exists())
        self.assertFalse((self.library.frames_root / 'one').exists())
        self.assertFalse(self.job._archive_source.exists())
        self.assertTrue(self.other._archive_source.exists())
        self.assertNotIn('one', S.STATE.jobs)
        self.assertNotIn('one', S.STATE.uploads)
        self.assertEqual([v['video_id'] for v in self.client.get('/api/videos').json()['videos']], ['two'])
        self.assertEqual(self.client.get('/api/videos/one/analysis').status_code, 404)
        self.assertEqual(self.client.delete('/api/videos/one').status_code, 404)

    def test_active_or_finishing_worker_cannot_be_deleted(self):
        self.job.state = 'running'
        self.assertEqual(self.client.delete('/api/videos/one').status_code, 409)
        self.job.state = 'done'
        self.job._detect_worker = Mock(is_alive=Mock(return_value=True))
        self.assertEqual(self.client.delete('/api/videos/one').status_code, 409)
        self.assertTrue(self.job._archive_source.exists())

    def test_qa_prevents_delete_without_losing_archive(self):
        with S.QA_BUSY:
            self.assertEqual(self.client.delete('/api/videos/one').status_code, 409)
        self.assertTrue(self.library.folder('one').exists())

    def test_delete_cannot_escape_frame_root(self):
        escaped = self.root / 'untouched'
        escaped.mkdir(); (escaped / 'keep.txt').write_text('keep')
        with patch.object(self.library, 'frames_root', self.root / 'frames/../untouched/child'):
            # Resolving .. is legal only within the configured root, never beyond.
            with self.assertRaises(ValueError):
                self.library.delete('../untouched')
        self.assertTrue((escaped / 'keep.txt').exists())
        self.assertTrue(self.job._archive_source.exists())

    def test_shared_or_unowned_source_is_retained(self):
        self.other._archive_source = self.job._archive_source
        self.library.save(self.other)
        result = self.client.delete('/api/videos/one').json()
        self.assertTrue(result['source_retained'])
        self.assertTrue(self.job._archive_source.exists())

    def test_archive_escape_is_rejected_before_media_removal(self):
        outside = self.root / 'outside'
        outside.mkdir()
        (outside / 'metadata.json').write_text(json.dumps(self.library.load('one')), encoding='utf-8')
        with patch.object(self.library, 'folder', return_value=outside), self.assertRaises(ValueError):
            self.library.delete('one')
        self.assertTrue((outside / 'metadata.json').exists())
        self.assertTrue(self.job._archive_source.exists())
        self.assertTrue((self.library.frames_root / 'one/f_00000.jpg').exists())

    def test_file_error_keeps_archive_and_releases_question_lock(self):
        with patch('webui.video_library.shutil.rmtree', side_effect=PermissionError('in use')):
            self.assertEqual(self.client.delete('/api/videos/one').status_code, 409)
        self.assertTrue(self.library.folder('one').exists())
        self.assertFalse(S.QA_BUSY.locked())

    def test_history_is_bounded_but_total_count_is_retained(self):
        for i in range(230):
            self.library.record(self.job, {'window_id': str(i), 'violations': {}})
        result = self.client.get('/api/videos/one/analysis').json()
        self.assertEqual(len(result['window_results']), 200)
        self.assertEqual(result['window_results'][0]['index'], 32)


class SpeechTests(unittest.TestCase):
    def test_inputs_are_bounded_and_voice_is_allowlisted(self):
        self.assertEqual(validate_speech({'text': ' 你好 ', 'voice': 'yunxi', 'rate': .9}), ('你好', 'yunxi', -10))
        for body in ({'text': ''}, {'text': 'x'*241}, {'text': []}, {'text': 'x', 'voice': []},
                     {'text': 'x', 'voice': 'file://something'}, {'text': 'x', 'rate': True},
                     {'text': 'x', 'rate': float('nan')}, {'text': 'x', 'rate': 10}):
            with self.subTest(body=body), self.assertRaises(ValueError): validate_speech(body)

    def test_audio_cached_and_failures_release_capacity(self):
        service = SpeechService()
        async def run():
            with patch.object(service, '_render', new=AsyncMock(return_value=b'mp3')) as render:
                self.assertEqual(await service.synthesize('你好', 'xiaoxiao', 0), b'mp3')
                self.assertEqual(await service.synthesize('你好', 'xiaoxiao', 0), b'mp3')
                self.assertEqual(render.await_count, 1)
            with patch.object(service, '_render', new=AsyncMock(side_effect=RuntimeError('offline'))):
                with self.assertRaises(RuntimeError): await service.synthesize('新文字', 'xiaoxiao', 0)
            self.assertTrue(service.slots.acquire(False)); self.assertTrue(service.slots.acquire(False))
            with self.assertRaises(BlockingIOError): await service.synthesize('满', 'xiaoxiao', 0)
            service.slots.release(); service.slots.release()
        asyncio.run(run())

    def test_api_returns_audio_and_explicit_fallback_errors(self):
        with TestClient(S.create_app()) as client:
            with patch.object(S.SPEECH, 'synthesize', new=AsyncMock(return_value=b'mp3')):
                response = client.post('/api/speech/synthesize', json={'text': '你好'})
                self.assertEqual(response.content, b'mp3')
                self.assertEqual(response.headers['content-type'], 'audio/wav')
                self.assertEqual(response.headers['cache-control'], 'no-store')
            with patch.object(S.SPEECH, 'synthesize', new=AsyncMock(side_effect=RuntimeError('provider-secret'))):
                response = client.post('/api/speech/synthesize', json={'text': '你好'})
                self.assertEqual(response.status_code, 503)
                self.assertNotIn('provider-secret', response.text)
            self.assertEqual(client.post('/api/speech/synthesize', json={'text': 123}).status_code, 400)


if __name__ == '__main__': unittest.main()
