"""Exercise repeated, changed and reordered frames at the actual feature boundary."""
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cluster.vision_cache import VisionCache


class FeatureCacheTests(unittest.TestCase):
    def make_cache(self, **kwargs):
        calls=[]
        def encode(pixels,grids,**kw):
            calls.append(len(grids))
            parts=torch.split(pixels,[int(g.prod()) for g in grids])
            pooled=tuple(p[:len(p)//4]*3 for p in parts)
            return types.SimpleNamespace(pooler_output=pooled,
                deepstack_features=[torch.cat(pooled)+10,torch.cat(pooled)+20])
        return VisionCache(encode,**kwargs),calls,encode

    def inputs(self, values):
        return torch.cat([torch.full((4,2),float(v)) for v in values]),torch.tensor([[1,2,2]]*len(values))

    def assert_features(self,a,b):
        for actual,expected in zip(a.pooler_output,b.pooler_output):
            torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        for actual,expected in zip(a.deepstack_features,b.deepstack_features):
            torch.testing.assert_close(actual,expected,rtol=0,atol=0)

    def test_overlap_order_and_duplicate_images_keep_exact_features(self):
        cache,calls,encode=self.make_cache()
        cache.keys=['a','b','a']
        pixels,grids=self.inputs([1,2,1])
        first=cache(pixels,grids)
        self.assertEqual(calls,[2])
        self.assert_features(first,encode(pixels,grids))
        cache.keys=['b','c','a','b']
        pixels,grids=self.inputs([2,3,1,2])
        second=cache(pixels,grids)
        self.assertEqual(calls[-1],1)
        self.assert_features(second,encode(pixels,grids))

    def test_expired_or_changed_geometry_recomputes(self):
        cache,calls,_=self.make_cache(ttl_sec=5)
        cache.keys=['a']
        pixels,grids=self.inputs([1])
        with patch('cluster.vision_cache.time.monotonic',return_value=0):cache(pixels,grids)
        with patch('cluster.vision_cache.time.monotonic',return_value=6):cache(pixels,grids)
        self.assertEqual(calls,[1,1])
        with patch('cluster.vision_cache.time.monotonic',return_value=7):
            cache(torch.ones((8,2)),torch.tensor([[1,2,4]]))
        self.assertEqual(calls,[1,1,1])

    def test_byte_limit_evicts_and_clear_releases_every_entry(self):
        cache,calls,_=self.make_cache(max_bytes=24)
        pixels,grids=self.inputs([1,2])
        cache.keys=['a','b']
        cache(pixels,grids)
        self.assertLessEqual(cache.bytes,24)
        self.assertEqual(len(cache.entries),1)
        cache.clear()
        self.assertEqual(cache.bytes,0)
        self.assertFalse(cache.entries)

    def test_unknown_output_falls_back_without_reusing_features(self):
        def legacy(pixels,grid,**kw):return ('legacy',len(grid))
        cache=VisionCache(legacy)
        cache.keys=['a','a']
        pixels,grids=self.inputs([1,1])
        self.assertEqual(cache(pixels,grids),('legacy',2))
        self.assertFalse(cache.supported)


if __name__=='__main__':unittest.main(verbosity=2)
