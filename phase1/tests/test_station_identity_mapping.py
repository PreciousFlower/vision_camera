"""Model evidence, display text and geometry share the ledger's manual identity."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from webui.station_identity import canonical_samples, canonical_review, canonical_window, canonical_alarm
from webui.workstation_runner import spatial_evidence


class StationIdentityMappingTests(unittest.TestCase):
    def test_manual_merge_maps_existing_buffer_geometry_before_model_request(self):
        original = [{'frame_id':'f_19', 'index':19, 'timestamp':246.9,
                     'tracks':[{'person_id':'P004', 'bbox':[.1,.1,.3,.9],
                                'pending_links':['P003','P001'], 'confidence':.9}]}]
        saved = copy.deepcopy(original)
        normalized = canonical_samples(original, {'P004':'P003','P003':'P001'})
        self.assertEqual(normalized[0]['tracks'][0]['person_id'], 'P001')
        self.assertEqual(normalized[0]['tracks'][0]['pending_links'], [])
        evidence = spatial_evidence(normalized[0], [{'id':'a','polygon':[[0,0],[1,0],[1,1],[0,1]]}], {'P001'})
        self.assertEqual(evidence['geometry_hint'][0]['person_id'], 'P001')
        self.assertEqual(original, saved)

    def test_delayed_result_maps_people_scene_and_reason_after_manual_merge(self):
        parsed = {'scene':'1号工位内有人员P004驻留，其余人员未在区域内。',
                  'observations':[{'person_id':'P004','frame_id':'f_19','station_id':'a',
                                   'state':'at_station','reason':'P004身体中心在蓝色区域'}],
                  'stations':[{'station_id':'a','frame_id':'f_19','visible':True,
                               'count':1,'reason':'P004在内，P003已离开'}]}
        original = copy.deepcopy(parsed)
        result = canonical_review(parsed, {'P004':'P003','P003':'P001'}, {'P001','P003','P004'})
        self.assertEqual(result['observations'][0]['person_id'], 'P001')
        self.assertIn('P001', result['scene'])
        self.assertNotIn('P004', result['scene'])
        self.assertNotIn('P003', result['stations'][0]['reason'])
        self.assertEqual(parsed, original)

    def test_model_cannot_confirm_fabricated_person_even_in_freeform_reason(self):
        result = canonical_review({'scene':'P999在岗','observations':[
            {'person_id':'P001','frame_id':'f_1','station_id':'a','state':'at_station','reason':'P999在区域内'}],
            'stations':[{'station_id':'a','frame_id':'f_1','visible':True,'count':1,'reason':'人员P999清晰可见'}]}, {}, {'P001'})
        self.assertEqual(result['observations'][0]['state'], 'uncertain')
        self.assertFalse(result['stations'][0]['visible'])
        self.assertIsNone(result['stations'][0]['count'])
        self.assertNotIn('P999', result['scene'])
        self.assertTrue(result['validation_warnings'])

    def test_simultaneous_people_are_not_merged_without_explicit_alias(self):
        samples = [{'tracks':[{'person_id':'P001','bbox':[0,0,.3,1]},
                               {'person_id':'P004','bbox':[.5,0,.9,1]}]}]
        normalized = canonical_samples(samples, {})
        self.assertEqual([t['person_id'] for t in normalized[0]['tracks']], ['P001','P004'])

    def test_old_window_display_maps_later_manual_link_but_preserves_raw(self):
        row = {'window_id':'job__w19', 'scene':'人员P004在岗',
               'people':[{'appearance':'P004','changed':'P004位于区域内'}],
               'raw':'{"scene":"人员P004在岗"}',
               'station_diagnostic':{'raw':'旧P004'},
               'station_review':{'scene':'人员P004在岗','observations':[
                   {'person_id':'P004','frame_id':'f_19','station_id':'a',
                    'state':'at_station','reason':'P004在区域内'}],'stations':[]}}
        original = copy.deepcopy(row)
        result = canonical_window(row, {'P004':'P001'}, {'P001'})
        self.assertEqual(result['people'][0]['appearance'], 'P001')
        self.assertEqual(result['station_review']['observations'][0]['person_id'], 'P001')
        self.assertEqual(result['identity_mapping'], [{'from':'P004','to':'P001'}])
        self.assertEqual(result['raw'], row['raw'])
        self.assertEqual(result['station_diagnostic'], row['station_diagnostic'])
        self.assertEqual(row, original)

    def test_alias_cycle_is_rejected_instead_of_hanging_request(self):
        with self.assertRaises(ValueError):
            canonical_samples([{'tracks':[{'person_id':'P004'}]}], {'P004':'P003','P003':'P004'})

    def test_history_links_deduplicate_people_and_keep_latest_frame_state(self):
        row = {'scene':'P004已返回，P003曾离开', 'frame_indices':[10,20], 'frame_times':[5,10],
               'people':[{'appearance':'P004'}, {'appearance':'P003'}],
               'station_review':{'observations':[
                   {'person_id':'P004','frame_id':'f_20','station_id':'a','state':'at_station','reason':'P004已返回'},
                   {'person_id':'P003','frame_id':'f_10','station_id':'a','state':'outside','reason':'P003曾离开'}],
                   'stations':[]}}
        result = canonical_window(row, {'P003':'P001','P004':'P001'}, {'P001'})
        self.assertEqual(len(result['people']), 1)
        self.assertEqual(result['people'][0]['appearance'], 'P001')
        self.assertEqual(result['people'][0]['action'], 'at_station')
        self.assertIn('已返回', result['people'][0]['changed'])

    def test_legacy_without_times_preserves_display_state_when_model_array_is_reversed(self):
        row = {'scene':'P004已返回', 'people':[{'appearance':'P004','action':'at_station','where':'a','changed':'P004已返回'}],
               'station_review':{'observations':[
                   {'person_id':'P004','frame_id':'f_20','station_id':'a','state':'at_station','reason':'P004已返回'},
                   {'person_id':'P003','frame_id':'f_10','station_id':'a','state':'outside','reason':'P003曾离开'}],
                   'stations':[]}}
        result = canonical_window(row, {'P003':'P001','P004':'P001'}, {'P001'})
        self.assertEqual(result['people'][0]['action'], 'at_station')
        self.assertEqual(result['people'][0]['where'], 'a')
        self.assertIn('已返回', result['people'][0]['changed'])

    def test_alarm_text_normalizes_explicit_link_without_changing_audit_status(self):
        alert = {'person_id':'P003','reason':'P003已离开工位','person_desc':'灰衣人员P003',
                 'latest_review_reason':'P003在区域外', 'retraction_reason':'P003已返回',
                 'evidences':['P003位于工位外'],'review_status':'confirmed',
                 'status_history':[{'reason':'P003当时在区域外'}]}
        original = copy.deepcopy(alert)
        result = canonical_alarm(alert, {'P003':'P001'})
        self.assertEqual(result['person_id'], 'P001')
        for field in ('reason','person_desc','latest_review_reason','retraction_reason'):
            self.assertIn('P001', result[field])
            self.assertNotIn('P003', result[field])
        self.assertEqual(result['evidences'], ['P001位于工位外'])
        self.assertEqual(result['original_reason'], alert['reason'])
        self.assertEqual(result['status_history'], alert['status_history'])
        self.assertEqual(result['review_status'], 'confirmed')
        self.assertEqual(alert, original)
        self.assertEqual(canonical_alarm(result, {'P003':'P001'}), result)

    def test_alarm_multiple_aliased_ids_retains_original_instead_of_inventing_conflicting_person(self):
        alert = {'person_id':'P004','reason':'P004在工位内，P003已离开',
                 'review_status':'confirmed','status_history':[{'reason':'P003曾离开'}]}
        result = canonical_alarm(alert, {'P003':'P001','P004':'P001'})
        self.assertEqual(result['person_id'], 'P001')
        self.assertEqual(result['reason'], alert['reason'])
        self.assertIn('reason', result['identity_collision_fields'])
        self.assertIn('保留当时原文', result['identity_note'])
        self.assertEqual(result['review_status'], 'confirmed')
        self.assertEqual(result['status_history'], alert['status_history'])


if __name__ == '__main__':
    unittest.main()
