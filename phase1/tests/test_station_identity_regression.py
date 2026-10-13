"""Anonymous identity survives observable returns without merging lookalike people."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from webui.person_tracking import PersonDetector, AnonymousTracker, _clothing_appearance, _appearance_distance


def person(box=(.35, .1, .55, .9), color=(1., 0., 0.)):
    return {'bbox': list(box), '_appearance': list(color), 'confidence': .9}


class StationIdentityRegressionTests(unittest.TestCase):
    def test_single_person_repeated_long_departures_does_not_create_nine_people(self):
        tracker = AnonymousTracker(max_age=3)
        identities = []
        for visit in range(9):
            start = visit * 40
            for timestamp in (start, start + .5, start + 1):
                identities.append(tracker.update(timestamp, [person()])[0]['person_id'])
            tracker.update(start + 5, [])
        self.assertEqual(set(identities), {'P001'})

    def test_long_return_at_different_location_uses_unique_clothing_not_distance(self):
        tracker = AnonymousTracker(max_age=3)
        first = tracker.update(0, [person((.02, .1, .2, .9))])[0]
        tracker.update(5, [])
        returned = tracker.update(20, [person((.75, .1, .95, .9))])[0]
        self.assertEqual(returned['person_id'], first['person_id'])

    def test_simultaneously_visible_people_with_same_clothes_keep_separate_ids(self):
        tracker = AnonymousTracker(max_age=3)
        first = tracker.update(0, [person((.05, .1, .25, .9)), person((.7, .1, .9, .9))])
        second = tracker.update(.5, [person((.69, .1, .89, .9)), person((.06, .1, .26, .9))])
        self.assertEqual(len({p['person_id'] for p in first}), 2)
        self.assertEqual([p['person_id'] for p in second], [first[1]['person_id'], first[0]['person_id']])

    def test_different_clothes_after_absence_are_not_merged(self):
        tracker = AnonymousTracker(max_age=3)
        first = tracker.update(0, [person()])[0]
        tracker.update(5, [])
        replacement = tracker.update(20, [person(color=(0., 1., 0.))])[0]
        self.assertNotEqual(replacement['person_id'], first['person_id'])

    def test_two_indistinguishable_absent_people_are_not_arbitrarily_merged(self):
        tracker = AnonymousTracker(max_age=3)
        first = tracker.update(0, [person((.05, .1, .25, .9)), person((.7, .1, .9, .9))])
        tracker.update(5, [])
        returned = tracker.update(20, [person()])[0]
        self.assertNotIn(returned['person_id'], {p['person_id'] for p in first})
        self.assertEqual(returned['identity_status'], 'pending')
        self.assertEqual(set(returned['pending_links']), {p['person_id'] for p in first})

    def test_absence_without_visual_descriptor_requires_new_identity(self):
        tracker = AnonymousTracker(max_age=3)
        first = tracker.update(0, [{'bbox': [.1, .1, .3, .9]}])[0]
        tracker.update(5, [])
        returned = tracker.update(20, [{'bbox': [.1, .1, .3, .9]}])[0]
        self.assertNotEqual(returned['person_id'], first['person_id'])

    def test_rapid_motion_does_not_fragment_unique_matching_clothes(self):
        tracker = AnonymousTracker(max_age=3)
        first = tracker.update(0, [person((.1, .1, .3, .9))])[0]
        moved = tracker.update(.5, [person((.4, .1, .6, .9))])[0]
        self.assertEqual(moved['person_id'], first['person_id'])

    def test_visible_edge_exit_has_evidence_but_static_edge_does_not(self):
        tracker = AnonymousTracker(max_age=3)
        tracker.update(0, [person((.7, .1, .9, .9))])
        exit_track = tracker.update(.5, [person((.8, .1, .99, .9))])[0]
        self.assertTrue(exit_track['departure_evidence'])
        static = tracker.update(1, [person((.8, .1, .99, .9))])[0]
        self.assertFalse(static['departure_evidence'])

    def test_return_near_boundary_does_not_invent_new_departure(self):
        tracker = AnonymousTracker(max_age=3)
        tracker.update(0, [person((.8, .1, .99, .9))])
        tracker.update(5, [])
        returned = tracker.update(20, [person((.02, .1, .22, .9))])[0]
        self.assertEqual(returned['person_id'], 'P001')
        self.assertFalse(returned['departure_evidence'])

    def test_pending_lookalike_return_does_not_mint_more_ids_every_frame(self):
        tracker = AnonymousTracker(max_age=3)
        tracker.update(0, [person((.05, .1, .25, .9)), person((.7, .1, .9, .9))])
        tracker.update(5, [])
        returns = [tracker.update(t, [person()])[0] for t in (20, 20.5, 21, 21.5)]
        self.assertEqual({p['person_id'] for p in returns}, {'P003'})
        self.assertTrue(all(p['identity_status']=='pending' for p in returns))

    def test_gallery_expiration_and_capacity_are_bounded(self):
        tracker = AnonymousTracker(max_age=1, gallery_age=10, gallery_limit=2)
        tracker.update(0, [person()])
        tracker.update(5, [person(color=(0., 1., 0.))])
        tracker.update(9, [person(color=(0., 0., 1.))])
        self.assertEqual(len(tracker.tracks), 2)
        tracker.update(25, [])
        self.assertEqual(tracker.tracks, {})

    def test_pose_samples_do_not_forget_first_visit_when_person_returns(self):
        tracker = AnonymousTracker(max_age=3)
        tracker.update(0, [person()])
        # Continuous body motion changes the central crop as arms, skin and
        # background enter it. The first garment sample remains a valid view;
        # it must not be discarded merely because more than eight poses arrive.
        for step in range(1, 21):
            mix = step / 20
            result = tracker.update(step * .5, [person(color=(1-mix, mix, 0.))])[0]
            self.assertEqual(result['person_id'], 'P001')
        tracker.update(15, [])
        returned = tracker.update(40, [person()])[0]
        self.assertEqual(returned['person_id'], 'P001')

    def test_manual_link_unifies_gallery_instead_of_recreating_old_track_id(self):
        tracker = AnonymousTracker(max_age=3)
        tracker.update(0, [person()])
        tracker.update(5, [])
        later = tracker.update(20, [person(color=(0., 1., 0.))])[0]
        self.assertEqual(later['person_id'], 'P002')
        tracker.link_identities({'P002':'P001'})
        tracker.update(25, [])
        returned = tracker.update(40, [person(color=(0., 1., 0.))])[0]
        self.assertEqual(returned['person_id'], 'P001')
        self.assertEqual(set(tracker.tracks), {'P001'})
        self.assertEqual(returned['identity_status'], 'manual')

    def test_gray_clothing_tolerates_webcam_white_balance_and_background_change(self):
        import numpy as np
        first = np.full((160, 90, 3), (173, 161, 159), dtype=np.uint8)
        returned = np.full((160, 90, 3), (155, 159, 167), dtype=np.uint8)
        returned[:, :15] = (40, 60, 120)
        tracker = AnonymousTracker(max_age=3)
        tracker.update(0, [person(color=_clothing_appearance(first))])
        tracker.update(5, [])
        result = tracker.update(30, [person(color=_clothing_appearance(returned))])[0]
        self.assertEqual(result['person_id'], 'P001')

    def test_actual_clothing_descriptor_separates_red_green_and_dark_gray(self):
        import numpy as np
        def descriptor(color):
            return _clothing_appearance(np.full((160,90,3),color,dtype=np.uint8))
        for left, right in (((30,30,190),(30,190,30)),((20,20,20),(180,180,180))):
            self.assertGreater(_appearance_distance(descriptor(left),descriptor(right)),.28)

    def test_invalid_or_changed_descriptor_does_not_break_tracking(self):
        tracker = AnonymousTracker()
        tracker.update(0, [person()])
        tracker.update(.5, [person(color=(1., 0., 0., 0.))])
        self.assertEqual(tracker.update(1,[person(color=(float('nan'),0.,1.))])[0]['person_id'],'P001')

    def test_low_confidence_background_person_does_not_enter_identity_gallery(self):
        import math
        import tempfile
        from types import SimpleNamespace
        from unittest.mock import patch
        import numpy as np
        output = np.zeros((1, 8400, 85), dtype=np.float32)
        # Observed webcam false positive: small background object at .434;
        # the visible foreground person's detector confidence is .90.
        for i, (center, size, objectness, score) in enumerate((
                ((100,240),(200,400),.95,.95),
                ((430,335),(65,82),.7,.62))):
            output[0,i,:4] = (center[0]/8-i, center[1]/8, math.log(size[0]/8), math.log(size[1]/8))
            output[0,i,4:6] = (objectness,score)
        class FakeSession:
            def __init__(self,*args,**kwargs):pass
            def get_inputs(self):return [SimpleNamespace(name='images')]
            def run(self,*args):return [output]
        fake_ort = SimpleNamespace(SessionOptions=SimpleNamespace,InferenceSession=FakeSession)
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules,{'onnxruntime':fake_ort}):
            path=Path(directory)/'weights.onnx';path.write_bytes(b'isolated inference output fixture')
            detector=PersonDetector(path)
            detections=detector.detect(np.full((480,640,3),160,dtype=np.uint8))
        self.assertEqual(len(detections),1)
        self.assertGreater(detections[0]['confidence'],.8)


if __name__ == '__main__':
    unittest.main()
