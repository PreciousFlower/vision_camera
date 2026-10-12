"""Service lifecycle regressions with fake SSH, GPU, tunnel and HTTP."""
from pathlib import Path
import sys
import time
import types
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fastapi.testclient import TestClient
from webui import cluster_client as C
from webui import server as S


class DeferredThread:
    threads = []

    def __init__(self, *, target, **kwargs):
        self.target = target
        self.started = False
        self.finished = False
        self.threads.append(self)

    def start(self):
        self.started = True

    def is_alive(self):
        return self.started and not self.finished

    def run(self):
        try:
            self.target()
        finally:
            self.finished = True


class ServiceLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.saved = S.STATE.session
        S.STATE.session = None
        DeferredThread.threads = []
        self.client = Mock()
        self.client.submit.return_value = "fixture-job"
        self.client.job_state.return_value = ("RUNNING", "fixture-node")
        self.client.list_my_jobs.return_value = []
        self.tunnel = Mock()
        self.tunnel.start.return_value = 12345
        self.tunnel.base_url.return_value = "http://fixture.invalid"
        self.http = Mock()
        self.http.json.return_value = {"ready": True, "model": "fixture", "device": "cpu"}
        self.patches = [patch.object(C, "Tunnel", return_value=self.tunnel),
                        patch.object(C, "time", types.SimpleNamespace(time=time.time,
                            sleep=Mock(side_effect=AssertionError("cancelled start entered wait")))),
                        patch("requests.get", return_value=self.http),
                        patch.object(S, "ensure_current_runtime"),
                        patch.object(S.STATE, "client", return_value=self.client),
                        patch.object(C, "deploy_service_package"),
                        patch.object(S, "threading", types.SimpleNamespace(Thread=DeferredThread))]
        for item in self.patches:
            item.start()
        self.api = TestClient(S.create_app())
        self.api.__enter__()

    def tearDown(self):
        self.api.__exit__(None, None, None)
        for item in reversed(self.patches):
            item.stop()
        S.STATE.session = self.saved

    def session(self):
        return C.ServiceSession(self.client, model="fixture")

    def start_cancelled(self, session):
        try:
            session.start()
        except C.ClusterError:
            pass
        self.assertEqual(session.state, "stopped")

    def test_duplicate_api_start_reserves_before_worker_dispatch(self):
        first = self.api.post("/api/service/start", json={})
        session = S.STATE.session
        second = self.api.post("/api/service/start", json={})
        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 400)
        self.assertIs(S.STATE.session, session)
        self.assertEqual(len(DeferredThread.threads), 1)
        self.client.submit.assert_not_called()

    def test_failed_start_thread_can_retry(self):
        class FailingThread(DeferredThread):
            def start(self):
                raise RuntimeError("thread unavailable")
        with patch.object(S, "threading", types.SimpleNamespace(Thread=FailingThread)):
            failed = self.api.post("/api/service/start", json={})
        self.assertEqual(failed.status_code, 500)
        self.assertEqual(S.STATE.session.state, "error")
        self.assertEqual(self.api.post("/api/service/start", json={}).status_code, 200)
        self.client.submit.assert_not_called()

    def test_cancelled_worker_must_exit_before_replacement_start(self):
        self.api.post("/api/service/start", json={})
        previous = S.STATE.session
        worker = DeferredThread.threads[-1]
        self.api.post("/api/service/stop")
        self.assertEqual(self.api.post("/api/service/start", json={}).status_code, 400)
        self.assertIs(S.STATE.session, previous)
        worker.run()
        self.assertEqual(self.api.post("/api/service/start", json={}).status_code, 200)

    def test_stop_during_upload_prevents_later_submit(self):
        self.assertEqual(self.api.post("/api/service/start", json={}).status_code, 200)
        session = S.STATE.session
        with patch.object(C, "deploy_service_package", side_effect=lambda *args:self.api.post("/api/service/stop")):
            DeferredThread.threads[-1].run()
        self.client.submit.assert_not_called()
        self.assertEqual(session.state, "stopped")

    def test_stop_before_worker_dispatch_prevents_upload_or_submit(self):
        self.api.post("/api/service/start", json={})
        session = S.STATE.session
        self.assertEqual(self.api.post("/api/service/stop").status_code, 200)
        with patch.object(C, "deploy_service_package") as upload:
            DeferredThread.threads[-1].run()
        upload.assert_not_called()
        self.client.submit.assert_not_called()
        self.assertEqual(session.state, "stopped")

    def test_stop_while_submit_returns_cancels_new_job(self):
        session = self.session()
        def submit(*args, **kwargs):
            session.stop()
            return "fixture-job"
        self.client.submit.side_effect = submit
        self.start_cancelled(session)
        self.client.cancel.assert_called_with("fixture-job")
        self.tunnel.start.assert_not_called()

    def test_stop_during_queue_poll_does_not_open_tunnel(self):
        session = self.session()
        def state(*args):
            session.stop()
            return "RUNNING", "fixture-node"
        self.client.job_state.side_effect = state
        self.start_cancelled(session)
        self.client.cancel.assert_called_with("fixture-job")
        self.tunnel.start.assert_not_called()

    def test_stop_while_tunnel_start_returns_does_not_load_model(self):
        session = self.session()
        def tunnel_start(*args):
            session.stop()
            return 12345
        self.tunnel.start.side_effect = tunnel_start
        with patch("requests.get") as health:
            self.start_cancelled(session)
            health.assert_not_called()
        self.tunnel.stop.assert_called()

    def test_stop_during_health_poll_cannot_revive_ready_state(self):
        session = self.session()
        def health(*args, **kwargs):
            session.stop()
            return self.http
        with patch("requests.get", side_effect=health):
            self.start_cancelled(session)
        self.client.cancel.assert_called_with("fixture-job")
        self.assertIsNone(session.tunnel)

    def test_cancelled_inflight_operation_errors_are_not_startup_failures(self):
        for stage in ("submit", "job_state", "tunnel", "health"):
            with self.subTest(stage=stage):
                session = self.session()
                def fail(*args, **kwargs):
                    session.stop()
                    raise RuntimeError("operation closed during cancellation")
                if stage == "health":
                    target = patch("requests.get", side_effect=fail)
                else:
                    owner = self.tunnel if stage == "tunnel" else self.client
                    name = "start" if stage == "tunnel" else stage
                    target = patch.object(owner, name, side_effect=fail)
                with target:
                    self.start_cancelled(session)
                self.assertIsNone(session.tunnel)

    def test_upload_error_after_stop_preserves_cancelled_state(self):
        self.api.post("/api/service/start", json={})
        session = S.STATE.session
        def upload(*args):
            self.api.post("/api/service/stop")
            raise RuntimeError("upload connection closed")
        with patch.object(C, "deploy_service_package", side_effect=upload):
            DeferredThread.threads[-1].run()
        self.client.submit.assert_not_called()
        self.assertEqual(session.state, "stopped")

    def test_failed_release_keeps_job_id_and_error_visible(self):
        self.api.post("/api/service/start", json={})
        session = S.STATE.session
        def state(*args):
            session._stop_event.set()
            raise RuntimeError("poll closed")
        self.client.job_state.side_effect = state
        self.client.cancel.side_effect = RuntimeError("cancel denied")
        DeferredThread.threads[-1].run()
        self.assertEqual(session.state, "error")
        self.assertEqual(session.jobid, "fixture-job")
        self.assertIn("cancel denied", session.message)
        self.assertIn("释放计算节点失败", session.message)

    def test_normal_start_stop_keeps_contract_and_orphan_sweep(self):
        session = self.session()
        session.start()
        self.assertEqual(session.state, "ready")
        self.assertEqual(session.jobid, "fixture-job")
        self.assertEqual(session.base_url(), "http://fixture.invalid")
        self.client.list_my_jobs.return_value = [{"name": "civility-serve", "jobid": "orphan"},
                                                {"name": "other-job", "jobid": "preserved"}]
        cancelled = session.stop()
        self.assertEqual(cancelled, ["fixture-job", "orphan"])
        self.assertEqual(session.state, "stopped")
        self.assertIsNone(session.tunnel)
        self.assertEqual([call.args[0] for call in self.client.cancel.call_args_list],
                         ["fixture-job", "orphan"])

    def test_repeated_stop_after_release_updates_state_and_allows_restart(self):
        session = self.session()
        session.state = 'ready'
        session.jobid = 'fixture-job'
        session.node = 'fixture-node'
        session.tunnel = self.tunnel
        S.STATE.session = session
        with TestClient(S.create_app(), raise_server_exceptions=False) as api:
            self.assertEqual(api.post('/api/service/stop').status_code, 200)
            # Slurm rejects a second cancellation after the job was released.
            self.client.cancel.side_effect = RuntimeError('invalid job id')
            repeated = api.post('/api/service/stop')
            state = api.get('/api/state').json()['service']
            self.assertEqual(state['state'], 'stopped', state)
            self.assertEqual(repeated.status_code, 200)
            self.assertEqual(self.client.cancel.call_count, 1)
            self.assertIsNone(state['jobid'])
            self.assertIsNone(state['node'])
            self.assertEqual(api.post('/api/service/start', json={}).status_code, 200)

    def test_release_error_is_retryable_and_cannot_allocate_another_node(self):
        session = self.session(); session.state = 'ready'; session.jobid = 'fixture-job'
        S.STATE.session = session
        self.client.cancel.side_effect = RuntimeError('offline')
        failed = self.api.post('/api/service/stop')
        self.assertEqual(failed.status_code, 503)
        state = self.api.get('/api/state').json()['service']
        self.assertEqual(state['state'], 'error')
        self.assertTrue(state['release_pending'])
        self.assertEqual(state['jobid'], 'fixture-job')
        self.assertEqual(self.api.post('/api/service/start', json={}).status_code, 409)
        self.client.cancel.side_effect = None
        self.assertEqual(self.api.post('/api/service/stop').status_code, 200)
        self.assertFalse(self.api.get('/api/state').json()['service']['release_pending'])
        self.assertEqual(self.api.post('/api/service/start', json={}).status_code, 200)


class SchedulerReleaseTests(unittest.TestCase):
    def client(self):
        return C.ClusterClient('/fixture', 'python', '/models')

    def test_release_ack_and_already_absent_are_accepted(self):
        client = self.client()
        with patch.object(client, 'run', return_value=(0, '', '')) as run:
            client.cancel('12345'); self.assertEqual(run.call_count, 1)
        with patch.object(client, 'run', side_effect=[(1, '', 'invalid job id'), (0, '98765\n', '')]):
            client.cancel('12345')

    def test_network_and_scheduler_failure_cannot_claim_release(self):
        client = self.client()
        with patch.object(client, 'run', side_effect=RuntimeError('offline')):
            with self.assertRaises(C.ClusterError): client.cancel('12345')
        with patch.object(client, 'run', side_effect=[(1, '', 'denied'), (0, '12345\n', '')]):
            with self.assertRaises(C.ClusterError): client.cancel('12345')
        with patch.object(client, 'run', side_effect=[(1, '', 'denied'), (1, '', 'failed')]):
            with self.assertRaises(C.ClusterError): client.cancel('12345')
        with patch.object(client, 'run', return_value=(1, '', 'failed')):
            with self.assertRaises(C.ClusterError): client.list_my_jobs()

    def test_invalid_id_never_runs_a_shell_command(self):
        client = self.client()
        with patch.object(client, 'run') as run:
            with self.assertRaises(C.ClusterError): client.cancel('123;unsafe')
            run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
