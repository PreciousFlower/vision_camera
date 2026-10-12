"""Approved audit fixes and human review workflow, isolated from user media/GPU."""
import asyncio
import copy
import io
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fastapi import HTTPException, UploadFile
from fastapi.testclient import TestClient
from webui import server as S
from webui.cluster_client import ServiceSession
from webui.video_library import RevisionConflict
from webui.uploads import CHUNK_BYTES, save_upload, UploadGuard
from webui.alert_feedback import FeedbackStore, evidence_frames


class EvidenceFeedbackTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.saved = S.STATE.cfg, S.STATE.jobs, S.STATE.uploads, S.STATE.session
        S.STATE.cfg = copy.deepcopy(S.STATE.cfg)
        S.STATE.cfg['paths'].update({key: str(self.root / key) for key in ('data_dir', 'frames_dir', 'results_dir')})
        S.STATE.jobs, S.STATE.uploads, S.STATE.session = {}, {}, None
        self.runtime = patch.object(S, 'ensure_current_runtime'); self.runtime.start()
        self.client = TestClient(S.create_app())
        self.library = S.video_library()
        self.job = S.JobRecord('audit_video', 'fixture.mp4', state='done', windows_seen=3)
        self.library.uploads_root.mkdir(parents=True)
        self.job._archive_source = self.library.uploads_root / 'shared.mp4'
        self.job._archive_source.write_bytes(b'original')
        self.row = {'window_id': 'w1', 't_start': 2, 't_end': 6, 'frames': 2, 'frame0': 6,
                    'frame_indices': [6, 12], 'frame_labels': ['t=2.0s', 't=4.0s'], 'frame_times': [2.125, 4.125],
                    'scene': '丢弃后离开', 'violations': {'littering': {'violation': True, 'confidence': .95, 'evidence': {'dropped': 2}}}}
        self.job.alerts = [{'behavior_id': 'littering', 'window_ids': ['w1'], 't_start': 2, 't_end': 6}]
        self.library.begin(self.job)
        for i in range(3):
            self.library.record(self.job, self.row | {'window_id': 'w'+str(i)})
        self.library.save(self.job)
        self.revision = self.job._analysis_revision
        directory = self.library.frames_root / self.job.job_id
        directory.mkdir(parents=True)
        (directory / 'f_00006.jpg').write_bytes(b'frame6')
        (directory / 'f_00012.jpg').write_bytes(b'frame12')

    def tearDown(self):
        self.client.close(); self.runtime.stop()
        S.STATE.cfg, S.STATE.jobs, S.STATE.uploads, S.STATE.session = self.saved
        self.temp.cleanup()

    def feedback(self, **extra):
        return self.client.post('/api/feedback/audit_video', json={'window_id': 'w1', 'analysis_revision': self.revision,
                         'behavior_id': 'littering', 'label': 'false_positive', 'frame_id': 'f_12', 'note': '测试备注', **extra})

    def test_evidence_links_neighbor_windows_and_exact_archived_times(self):
        result = self.client.get('/api/evidence/audit_video?window_id=w1&behavior_id=littering').json()
        self.assertEqual({f['window_id'] for f in result['frames']}, {'w0', 'w1', 'w2'})
        self.assertEqual(result['frames'][0]['time_sec'], 2.125)
        self.assertNotIn(str(self.root), str(result))
        frame = next(f for f in result['frames'] if f['is_anchor'])
        self.assertEqual(self.client.get(frame['image_url']).content, b'frame6')
        self.assertTrue(result['source_available'])

    def test_feedback_persists_updates_and_preserves_model_judgment(self):
        before = self.library.windows('audit_video')
        response = self.feedback()
        self.assertEqual(response.status_code, 200, response.text)
        first = response.json()['feedback']
        self.assertEqual(first['frame']['time_sec'], 4.125)
        self.assertTrue(first['model_result']['violation'])
        self.assertEqual(self.feedback(label='correct').status_code, 200)
        fresh_store = FeedbackStore(S.feedback_store().root)
        records = fresh_store.list('audit_video')
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]['label'], 'correct')
        self.assertEqual(first['feedback_id'], records[0]['feedback_id'])
        self.assertEqual(before, self.library.windows('audit_video'))
        self.assertEqual(self.client.get('/api/feedback/audit_video').json()['feedback'], records)

    def test_feedback_rejects_stale_revision_foreign_frames_and_invalid_fields(self):
        for fields, code in [({'analysis_revision': 'stale'}, 409), ({'frame_id': 'f_999'}, 400),
                             ({'behavior_id': 'unknown'}, 400), ({'note': 'x'*1001}, 400),
                             ({'label': []}, 400), ({'window_id': 'missing'}, 404)]:
            response = self.feedback(**fields)
            self.assertEqual(response.status_code, code, response.text)
        self.assertEqual(S.feedback_store().list('audit_video'), [])

    def test_missed_label_and_risk_feedback_are_supported(self):
        self.assertEqual(self.feedback(behavior_id='smoking', frame_id=None, label='missed').status_code, 200)
        self.assertEqual(self.feedback(behavior_id='smoking', frame_id=None, label='false_positive').status_code, 400)
        row = self.row | {'risks': [{'risk_id': 'fire', 'name': '明火', 'alert': True}]}
        self.library.begin(self.job); self.library.record(self.job, row); self.library.save(self.job)
        self.revision = self.job._analysis_revision
        self.assertEqual(self.feedback(behavior_id='ai_risk:fire').status_code, 200)
        self.assertEqual(len(S.feedback_store().list('audit_video')), 2)

    def test_deleted_video_and_feedback_cannot_reappear(self):
        self.assertEqual(self.feedback().status_code, 200)
        self.assertEqual(self.client.delete('/api/videos/audit_video').status_code, 200)
        self.assertEqual(self.client.get('/api/feedback/audit_video').json()['feedback'], [])
        with self.assertRaises(FileNotFoundError):
            self.library._frame_from_row('audit_video', self.row, 6)
        self.assertFalse((self.library.frames_root / 'audit_video').exists())
        self.assertTrue(self.job._archive_source.exists())

    def test_failed_media_delete_restores_feedback_for_retry(self):
        self.assertEqual(self.feedback().status_code, 200)
        with patch('webui.video_library.shutil.rmtree', side_effect=PermissionError('in use')):
            self.assertEqual(self.client.delete('/api/videos/audit_video').status_code, 409)
        self.assertEqual(len(S.feedback_store().list('audit_video')), 1)
        self.assertTrue(self.library.folder('audit_video').exists())

    def test_restore_and_delete_serialize_without_recreating_cache(self):
        (self.library.frames_root / 'audit_video/f_00006.jpg').unlink()
        entered, resume = threading.Event(), threading.Event()
        original = self.library.source
        def source(metadata):
            entered.set(); self.assertTrue(resume.wait(3)); return original(metadata)
        def recover(command, **kwargs): Path(command[-1]).write_bytes(b'restored')
        failures = []
        def frame():
            try: self.library.frame('audit_video', 'f_6')
            except Exception as exc: failures.append(exc)
        with patch.object(self.library, 'source', side_effect=source), patch('webui.video_library.subprocess.run', side_effect=recover):
            reader = threading.Thread(target=frame); reader.start(); self.assertTrue(entered.wait(3))
            deletion = threading.Thread(target=self.library.delete, args=('audit_video',)); deletion.start()
            self.assertTrue(deletion.is_alive()); resume.set(); reader.join(3); deletion.join(3)
            self.assertFalse(reader.is_alive()); self.assertFalse(deletion.is_alive())
        self.assertEqual(failures, [])
        self.assertFalse((self.library.frames_root / 'audit_video').exists())

    def test_mixed_revision_history_returns_conflict_not_broken_urls(self):
        original = self.library.windows
        def replace(video_id):
            self.library.begin(self.job); self.library.record(self.job, self.row | {'window_id': 'new'})
            return original(video_id)
        with patch.object(self.library, 'windows', side_effect=replace):
            self.assertEqual(self.client.get('/api/videos/audit_video/analysis').status_code, 409)

    def test_upload_survives_refresh_and_backend_memory_reset_without_counting_as_work(self):
        response = self.client.post('/api/upload', files={'file': ('video.mp4', b'fake fixture bytes', 'video/mp4')})
        self.assertEqual(response.status_code, 200)
        job_id = response.json()['job_id']
        self.assertFalse(S._active_job())
        self.assertTrue(self.client.get('/api/job/'+job_id).json()['awaiting_analysis'])
        S.STATE.jobs.clear(); S.STATE.uploads.clear()
        self.assertTrue(self.client.get('/api/job/'+job_id).json()['awaiting_analysis'])
        with patch.object(S, 'threading', Mock(Thread=Mock(return_value=Mock()))):
            self.assertEqual(self.client.post('/api/detect/'+job_id, json={'use_mock': True}).status_code, 200)
        self.assertTrue(S._active_job())

    def test_upload_api_limits_files_and_body_and_leaves_no_partial_task(self):
        S.STATE.cfg['upload'] = {'max_bytes': 8}
        for name, data, code in [('big.mp4', b'123456789', 413), ('empty.mp4', b'', 400), ('script.exe', b'1', 400)]:
            response = self.client.post('/api/upload', files={'file': (name, data)})
            self.assertEqual(response.status_code, code, response.text)
        self.assertEqual(S.STATE.jobs, {})
        self.assertFalse(list(self.library.uploads_root.glob('*.part')))
        response = self.client.post('/api/upload', content=b'x', headers={'content-length': str(CHUNK_BYTES+100)})
        self.assertEqual(response.status_code, 413)
        prefix = b'--upload\r\nContent-Disposition: form-data; name="file"; filename="big.mp4"\r\nContent-Type: video/mp4\r\n\r\n'
        response = self.client.post('/api/upload', content=iter([prefix, b'x'*(CHUNK_BYTES+9)]),
                                    headers={'content-type': 'multipart/form-data; boundary=upload'})
        self.assertEqual(response.status_code, 413, response.text)
        self.assertFalse(list(self.library.uploads_root.glob('*.part')))

    def test_failed_health_is_reflected_and_recovers_without_new_job(self):
        import requests
        session = ServiceSession(Mock(), 'fixture')
        session.state = 'ready'; session.tunnel = Mock(alive=True); session.tunnel.base_url.return_value = 'http://fixture'
        S.STATE.session = session
        with patch('requests.get', side_effect=requests.ConnectionError('offline')):
            self.assertFalse(session.health()['ready'])
        self.assertFalse(S.service_is_ready(session))
        with patch.object(session, 'poll_health'):
            self.assertEqual(self.client.get('/api/state').json()['service']['state'], 'unreachable')
        with patch('requests.get', return_value=Mock(status_code=200, json=Mock(return_value={'ready': True}))):
            self.assertTrue(session.health()['ready'])
        self.assertEqual(session.status()['state'], 'ready')

    def test_inflight_health_cannot_revive_stopped_session(self):
        session = ServiceSession(Mock(), 'fixture'); session.state = 'ready'; session.tunnel = Mock(alive=True)
        def stop_inflight(*args, **kwargs):
            session.stop(sweep_orphans=False)
            return Mock(status_code=200, json=Mock(return_value={'ready': True}))
        with patch('requests.get', side_effect=stop_inflight):
            session.health()
        self.assertEqual(session.status()['state'], 'stopped')
        self.assertIsNone(session._health_result)

    def test_pending_upload_does_not_prevent_cost_guard_release(self):
        saved = S.STATE.last_visible_at, S.STATE.page_visible
        try:
            S.STATE.jobs['pending'] = S.JobRecord('pending', 'pending.mp4')
            S.STATE.session = Mock(state='ready', started_at=0)
            S.STATE.session.started_at = 1
            S.STATE.last_visible_at = 1; S.STATE.page_visible = False
            with patch.object(S, '_auto_release', return_value='released') as release:
                self.assertEqual(S.watchdog_tick(), 'released')
                release.assert_called_once()
        finally: S.STATE.last_visible_at, S.STATE.page_visible = saved

    def test_live_evidence_and_feedback_are_versioned_without_original_recording(self):
        job = S.JobRecord('live', '摄像头', is_live=True, state='running', source_kind='local')
        job._analysis_revision = 'live-revision'; job.window_results = [self.row]
        S.STATE.jobs['live'] = job
        result = self.client.get('/api/evidence/live?window_id=w1').json()
        self.assertTrue(result['is_live']); self.assertIsNone(result['source_url'])
        self.assertIn('revision=live-revision', result['frames'][0]['image_url'])
        response = self.client.post('/api/feedback/live', json={'analysis_revision': 'live-revision', 'window_id': 'w1',
                                   'behavior_id': 'littering', 'label': 'false_positive'})
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()['feedback']['is_live'])
        job._analysis_revision = 'new-revision'
        self.assertEqual(self.client.get(result['frames'][0]['image_url']).status_code, 409)

    def test_missing_worker_after_restart_restores_file_instead_of_hanging_stream(self):
        response = self.client.post('/api/upload', files={'file': ('pending.mp4', b'fixture')})
        job_id = response.json()['job_id']; job = S.STATE.jobs[job_id]
        job.state = 'running'; job._archive_source = S.STATE.uploads[job_id]
        self.library.begin(job); self.library.save(job)
        S.STATE.jobs.clear(); S.STATE.uploads.clear()
        recovered = self.client.get('/api/job/'+job_id).json()
        self.assertTrue(recovered['awaiting_analysis']); self.assertFalse(recovered['worker_active'])


class UploadStreamingTests(unittest.TestCase):
    def test_unknown_size_stream_is_bounded_closed_and_cleaned_on_limit(self):
        async def run(root):
            file = UploadFile(io.BytesIO(b'x'*(CHUNK_BYTES+3)), filename='stream.mp4')
            with patch.object(file, 'read', wraps=file.read) as read:
                with self.assertRaises(HTTPException) as result:
                    await save_upload(file, root, CHUNK_BYTES+1)
                self.assertEqual(result.exception.status_code, 413)
                self.assertTrue(all(call.args == (CHUNK_BYTES,) for call in read.call_args_list))
            self.assertTrue(file.file.closed)
            self.assertEqual(list(root.glob('*')), [])
        with tempfile.TemporaryDirectory() as directory:
            asyncio.run(run(Path(directory)))

    def test_chunked_body_limit_and_concurrency_capacity_are_enforced_before_parser(self):
        async def run():
            async def downstream(scope, receive, send):
                try: await receive()
                except HTTPException as exc: self.assertEqual(exc.status_code, 413)
            guard = UploadGuard(downstream, lambda: 8)
            scope = {'type': 'http', 'path': '/api/upload', 'headers': []}
            async def receive(): return {'type': 'http.request', 'body': b'x'*(CHUNK_BYTES+9)}
            messages = []
            async def send(message): messages.append(message)
            await guard(scope, receive, send)
            self.assertTrue(guard.slots.acquire(False)); self.assertTrue(guard.slots.acquire(False))
            await guard(scope, receive, send)
            self.assertEqual(messages[0]['status'], 429)
            guard.slots.release(); guard.slots.release()
        asyncio.run(run())


if __name__ == '__main__': unittest.main()
