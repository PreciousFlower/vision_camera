"""Frame references cannot discard an otherwise valid visual workstation review."""
import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from civility.station_review import build_messages, validate_result
from webui.cluster_client import ClusterError, ServiceSession


REQUEST = {
    'evidence': [{'frame_id': 'f_1', 'time_sec': 10.0}, {'frame_id': 'f_40', 'time_sec': 30.0}],
    'people': [{'person_id': 'P001'}],
    'stations': [{'id': 'a'}],
}
VALID = {
    'observations': [{'person_id': 'P001', 'station_id': 'a', 'state': 'at_station', 'frame_id': 'f_40', 'reason': '画面显示一人在区域内'}],
    'stations': [{'station_id': 'a', 'frame_id': 'f_40', 'visible': True, 'count': 1, 'reason': '区域清晰可见一人'}],
}


class StationProtocolRegressionTests(unittest.TestCase):
    def test_unique_equivalent_frame_reference_is_normalized(self):
        for reference in ('f_040', '40', 40, 'frame_id=f_40', 'f_40 @ 30.000s'):
            with self.subTest(reference=reference):
                result = copy.deepcopy(VALID)
                result['stations'][0]['frame_id'] = reference
                reviewed = validate_result(result, REQUEST)
                self.assertEqual(reviewed['stations'][0]['frame_id'], 'f_40')

    def test_bad_station_reference_does_not_destroy_same_window_valid_evidence(self):
        result = copy.deepcopy(VALID)
        result['stations'].append({**result['stations'][0], 'frame_id': 'f_999'})
        reviewed = validate_result(result, REQUEST)
        self.assertEqual(reviewed['stations'], VALID['stations'])
        self.assertTrue(reviewed['validation_warnings'])

    def test_all_unknown_references_never_become_confirmed_evidence(self):
        result = copy.deepcopy(VALID)
        for item in result['stations'] + result['observations']:
            item['frame_id'] = 'f_999'
        with self.assertRaises(ValueError):
            validate_result(result, REQUEST)

    def test_bad_person_entry_is_dropped_without_inventing_identity(self):
        result = copy.deepcopy(VALID)
        result['observations'].append({**result['observations'][0], 'person_id': 'P999'})
        reviewed = validate_result(result, REQUEST)
        self.assertEqual(reviewed['observations'], VALID['observations'])
        self.assertTrue(reviewed['validation_warnings'])

    def test_missing_count_or_wrong_count_is_unknown_not_zero(self):
        for count in (None, True, '0', -1, 999):
            result = copy.deepcopy(VALID)
            result['stations'][0]['count'] = count
            reviewed = validate_result(result, REQUEST)
            self.assertFalse(reviewed['stations'][0]['visible'])
            self.assertIsNone(reviewed['stations'][0]['count'])
            self.assertEqual(reviewed['observations'], VALID['observations'])

    def test_empty_scene_description_cannot_override_model_uncertainty(self):
        # Real Qwen trace: an empty chair was described correctly, but region
        # observability was false. Repair the prompt; never flip it by keywords.
        result = {'scene':'区域只有椅子和柜子，无人员可见','observations':[],
                  'stations':[{'station_id':'a','visible':False,'count':None,'frame_id':'f_40',
                               'reason':'区域含储物柜及椅子，无人员身体中心在内'}]}
        checked = validate_result(result, REQUEST)
        self.assertFalse(checked['stations'][0]['visible'])
        self.assertIsNone(checked['stations'][0]['count'])

    def test_prompt_separates_empty_visible_region_from_unobservable_region(self):
        # Lock down both distinct visual cases that the real empty-chair trace
        # confused. Visibility must describe the ROI, rather than person presence.
        request = copy.deepcopy(REQUEST)
        request['stations'][0]['polygon'] = [[0,0],[1,0],[1,1],[0,1]]
        messages = build_messages(['image','image'], request)
        system = messages[0]['content']
        self.assertIn('stations.visible', system)
        self.assertIn('区域是否可观察', system)
        self.assertIn('无人不等于区域不可见', system)
        self.assertIn('{"visible":true,"count":0}', system)
        self.assertIn('{"visible":false,"count":null}', system)
        self.assertIn('实际遮挡', system)

    def test_actual_sampling_id_is_never_confused_with_image_position(self):
        result = copy.deepcopy(VALID)
        result['stations'][0]['frame_id'] = '2'
        result['observations'][0]['frame_id'] = '第2张图'
        with self.assertRaises(ValueError):
            validate_result(result, REQUEST)

    def test_ambiguous_and_multiple_frame_references_are_not_guessed(self):
        request = copy.deepcopy(REQUEST)
        request['evidence'].append({'frame_id': 'f_040', 'time_sec': 31})
        for reference in ('040', 40, 'f_40 或 f_1', 'f_999 与 f_40'):
            result = copy.deepcopy(VALID)
            result['observations'] = []
            result['stations'][0]['frame_id'] = reference
            with self.assertRaises(ValueError):
                validate_result(result, request)
        self.assertEqual(validate_result(VALID, request)['stations'][0]['frame_id'], 'f_40')

    def test_valid_person_evidence_survives_all_station_items_invalid(self):
        result = copy.deepcopy(VALID)
        result['stations'][0]['frame_id'] = 'f_999'
        reviewed = validate_result(result, REQUEST)
        self.assertEqual(reviewed['observations'], VALID['observations'])
        self.assertEqual(reviewed['stations'], [])

    def test_user_record_sampling_ids_are_not_reindexed_to_eight_images(self):
        request = copy.deepcopy(REQUEST)
        request['evidence'] = [{'frame_id': f'f_{i}', 'time_sec': 141.312+(i-244)/63*37.235}
                               for i in (244,253,262,271,280,289,298,307)]
        result = copy.deepcopy(VALID)
        result['observations'] = []
        result['stations'][0]['frame_id'] = 307
        self.assertEqual(validate_result(result, request)['stations'][0]['frame_id'], 'f_307')
        for reference in ('8', '第8帧', None):
            result['stations'][0]['frame_id'] = reference
            with self.assertRaises(ValueError):
                validate_result(result, request)

    def test_remote_and_local_revalidation_preserve_warnings_and_conflicts(self):
        result = copy.deepcopy(VALID)
        result['stations'].append({**result['stations'][0], 'count': 0})
        first = validate_result(result, REQUEST)
        self.assertFalse(first['stations'][0]['visible'])
        self.assertEqual(validate_result(first, REQUEST), first)


class StationClientRegressionTests(unittest.TestCase):
    def setUp(self):
        self.session = ServiceSession(None, 'test')
        self.session.base_url = lambda: 'http://model.invalid'
        self.stations = [{'id': 'a', 'polygon': [[0,0],[1,0],[1,1],[0,1]]}]
    def review(self, result):
        response = Mock()
        response.json.return_value = result
        response.raise_for_status.return_value = None
        with patch('requests.post', return_value=response) as post:
            try:
                return self.session.review_workstations(['image','image'], REQUEST['evidence'], self.stations, REQUEST['people'])
            finally:
                self.assertEqual(post.call_count, 1)
    def test_older_remote_schema_error_recovers_exact_raw_without_gpu_retry(self):
        import json
        output = copy.deepcopy(VALID)
        output['stations'][0]['frame_id'] = 40
        raw = json.dumps(output, ensure_ascii=False)
        result = self.review({'parsed':None, 'error':'Schema 校验失败: 工位画面引用无效', 'raw':raw, 'tokens_out':128})
        self.assertIsNone(result['error'])
        self.assertEqual(result['parsed']['stations'][0]['frame_id'], 'f_40')
        self.assertEqual(result['station_diagnostic']['raw'], raw)
        self.assertTrue(result['protocol_warnings'])
    def test_unusable_original_output_is_attached_to_failure(self):
        import json
        output = copy.deepcopy(VALID)
        for item in output['stations'] + output['observations']:item['frame_id'] = 'f_999'
        raw = json.dumps(output)
        with self.assertRaises(ClusterError) as caught:
            self.review({'parsed':None,'error':'Schema 校验失败: 工位画面引用无效','raw':raw,'finish_reason':'stop'})
        self.assertEqual(caught.exception.station_diagnostic['raw'], raw)
        self.assertEqual(caught.exception.station_diagnostic['request_frame_ids'], ['f_1','f_40'])
    def test_infrastructure_error_is_not_reclassified_as_valid_visual_review(self):
        with self.assertRaises(ClusterError):
            self.review({'parsed':VALID,'error':'CUDA out of memory','raw':'valid-looking data'})


if __name__ == '__main__':
    unittest.main()
