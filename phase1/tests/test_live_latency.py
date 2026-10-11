"""Protect live timing and the small-batch service configuration."""
import copy
from pathlib import Path
import sys
import unittest
import tempfile
import types
import threading
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from webui import server as S


class LiveLatencyTests(unittest.TestCase):
    def setUp(self):
        self.saved=copy.deepcopy(S.STATE.cfg)
        S.STATE.cfg['live'].update(window_stride_sec=2,continuous_batch_windows=2,
                                   frames_per_window=4,context_sec=0)

    def tearDown(self):S.STATE.cfg=self.saved

    def test_fast_pipeline_does_not_claim_growing_delay(self):
        job=S.JobRecord('timing','camera',is_live=True,continuous=True,
                        source_kind='local',source_spec='0',last_batch_sec=3.6)
        raw=[[(0,'f_00000.jpg'),(2,'f_00001.jpg')],
             [(2,'f_00002.jpg'),(4,'f_00003.jpg')]]
        with patch('webui.capture.capture_windows',return_value=raw), \
                patch.object(S.STATE,'log_line') as log:
            S.capture_live_windows(job,'0')
        self.assertFalse(job.behind,'3.6s processing fits into 4s of video')
        self.assertFalse(any('逐渐滞后' in str(c) for c in log.call_args_list))

    def test_slow_inference_takes_recent_frames_instead_of_old_prefetch(self):
        with tempfile.TemporaryDirectory() as directory:
            S.STATE.cfg['paths']['frames_dir']=directory
            S.STATE.cfg['live']['context_sec']=4
            history=[(100+i/8,b'frame-'+str(i).encode()) for i in range(161)]
            reader=types.SimpleNamespace(dead=False,snapshot_history=lambda:history)
            stream=types.SimpleNamespace(reader=reader,kind='local')
            job=S.JobRecord('fresh','camera',is_live=True,continuous=True,source_kind='local',
                capture_started_at=100,last_recent_wall=108,last_batch_sec=6)
            old=[[(6,'f_00000.jpg'),(8,'f_00001.jpg')]]
            with patch('webui.capture.capture_windows',return_value=old) as legacy, \
                    patch.object(S.time,'time',return_value=120.1), \
                    patch.object(S.STATE,'log_line'):
                windows=S.capture_live_windows(job,'0',stream)
            legacy.assert_not_called()
            self.assertEqual([w.t_end for w in windows],[18,20])
            self.assertTrue(all(len(w.frames)==12 for w in windows))
            self.assertLess(windows[0].t_start,windows[0].t_end-2)
            self.assertTrue(set(windows[0].frames)&set(windows[1].frames))
            self.assertTrue(job.behind,'Slow model must still report a throughput limit')
            self.assertEqual(S.STATE.cfg['live']['window_stride_sec'],2)

    def test_stale_capture_is_an_error_instead_of_a_repeated_picture(self):
        reader=types.SimpleNamespace(dead=False,snapshot_history=lambda:[(120,b'old')])
        job=S.JobRecord('stale','camera',is_live=True,continuous=True,source_kind='local',
                        capture_started_at=100,last_recent_wall=108)
        with patch.object(S.time,'time',return_value=125):
            with self.assertRaises(S.ClusterError):S.capture_recent_windows(job,reader)

    def test_continuous_job_dispatches_fresh_batches_after_slow_model(self):
        saved_jobs, saved_session = S.STATE.jobs, S.STATE.session
        clock = [104.0]
        requests = []
        barrier = threading.Barrier(2, timeout=3)
        lock = threading.Lock()
        history = [(100+i/8,b'frame-'+str(i).encode()) for i in range(33)]
        reader = types.SimpleNamespace(dead=False,snapshot_history=lambda:list(history))
        stream = types.SimpleNamespace(reader=reader,kind='local',closed=False,
            ensure=lambda:None,ensure_reader=lambda **kw:None)
        stream.close = lambda:setattr(stream,'closed',True)
        job = S.JobRecord('dispatch','camera',is_live=True,continuous=True,
                          source_kind='local',source_spec='0',capture_started_at=100)
        def infer(frames,labels,**kwargs):
            with lock:
                requests.append(labels)
                request_no=len(requests)
            barrier.wait()
            if request_no % 2 == 0:
                clock[0] += 6  # Model needs 6s; the two windows cover only 4s.
                history[:] = [(100+i/8,b'frame-'+str(i).encode())
                              for i in range(int((clock[0]-100)*8)+1)]
                if request_no == 4:job.stop_flag=True
            barrier.wait()
            return {'ok':True,'parsed':{'scene':'正常活动','people':[],
                'violations':{},'risks':[]},'raw':'{}','latency_sec':6,
                'roundtrip_sec':6,'error':None}
        try:
            with tempfile.TemporaryDirectory() as directory:
                S.STATE.cfg['paths']['frames_dir']=directory
                S.STATE.cfg['live'].update(context_sec=4,motion_gate=False)
                S.STATE.jobs={job.job_id:job}
                S.STATE.session=types.SimpleNamespace(state='ready',infer=infer)
                with patch.object(S,'LiveStream',return_value=stream), \
                        patch.object(S.time,'time',side_effect=lambda:clock[0]), \
                        patch.object(S.STATE,'log_line'):
                    S.run_job(job.job_id)
            self.assertEqual(sorted(float(labels[-1][2:-1]) for labels in requests),
                             [2,4,8,10])
            self.assertEqual(job.state,'stopped',job.error)
            self.assertTrue(stream.closed)
            self.assertTrue(all(result['result_age_sec']==6 for result in job.window_results
                                if result['t_end']==4))
        finally:
            S.STATE.jobs,S.STATE.session=saved_jobs,saved_session

    def test_reader_history_is_bounded_and_released_on_stop(self):
        from webui.capture import StreamReader
        reader=StreamReader(None,'local')
        reader.history_jpegs.extend((i,b'image') for i in range(300))
        self.assertEqual(len(reader.snapshot_history()),256)
        reader.stop()
        self.assertFalse(reader.snapshot_history())


if __name__=='__main__':unittest.main(verbosity=2)
