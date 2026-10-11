"""Bounded in-memory reuse of exact Qwen3-VL image features, never decisions.

Frame bytes, processing geometry and device must match. New frames still run
the vision encoder; every window still runs the language model and evidence gate.
"""
from collections import OrderedDict
import hashlib
from pathlib import Path
import time


def frame_key(frame):
    if frame.get('b64'):
        return hashlib.sha256(frame['b64'].encode('ascii')).hexdigest()
    if frame.get('path'):
        return hashlib.sha256(Path(frame['path']).read_bytes()).hexdigest()
    return None


class VisionCache:
    def __init__(self, get_features, *, max_bytes=512*1024*1024, ttl_sec=30):
        self.original=get_features
        self.max_bytes=max_bytes
        self.ttl_sec=ttl_sec
        self.keys=[]
        self.entries=OrderedDict()
        self.bytes=0
        self.hits=0
        self.misses=0
        self.supported=True

    def clear(self):
        self.entries.clear()
        self.keys=[]
        self.bytes=0

    def _expire(self, now):
        for key, (_, size, stored) in list(self.entries.items()):
            if now-stored>=self.ttl_sec:
                self.entries.pop(key)
                self.bytes-=size

    def __call__(self, pixel_values, image_grid_thw=None, **kwargs):
        import torch
        if (not self.supported or not self.keys or image_grid_thw is None or
                len(self.keys)!=len(image_grid_thw)):
            return self.original(pixel_values,image_grid_thw,**kwargs)
        now=time.monotonic()
        self._expire(now)
        grids=image_grid_thw.tolist()
        keys=[(key,tuple(grid),str(pixel_values.dtype),str(pixel_values.device))
              for key,grid in zip(self.keys,grids)]
        if any(key[0] is None for key in keys):
            return self.original(pixel_values,image_grid_thw,**kwargs)
        sizes=[int(t*h*w) for t,h,w in grids]
        if sum(sizes)!=len(pixel_values):
            return self.original(pixel_values,image_grid_thw,**kwargs)
        pixels=torch.split(pixel_values,sizes)
        needed=[]
        seen=set()
        resolved={key:self.entries[key][0] for key in keys if key in self.entries}
        for i,key in enumerate(keys):
            if key not in resolved and key not in seen:
                needed.append(i)
                seen.add(key)
        self.hits+=len(keys)-len(needed)
        self.misses+=len(needed)
        if needed:
            output=self.original(torch.cat([pixels[i] for i in needed]),
                                 image_grid_thw[needed],**kwargs)
            pooled=getattr(output,'pooler_output',None)
            deep=getattr(output,'deepstack_features',None)
            if not isinstance(pooled,(tuple,list)) or len(pooled)!=len(needed) or deep is None:
                self.supported=False
                self.clear()
                return self.original(pixel_values,image_grid_thw,**kwargs)
            merged_sizes=[len(image) for image in pooled]
            if any(len(layer)!=sum(merged_sizes) for layer in deep):
                self.supported=False
                self.clear()
                return self.original(pixel_values,image_grid_thw,**kwargs)
            layers=[torch.split(layer,merged_sizes) for layer in deep]
            for j,i in enumerate(needed):
                # Clone per image so a small entry cannot retain a whole batch's storage.
                value=(pooled[j].detach().clone(),[layer[j].detach().clone() for layer in layers],type(output))
                size=sum(t.numel()*t.element_size() for t in [value[0]]+value[1])
                resolved[keys[i]]=value
                if size<=self.max_bytes:
                    while self.entries and self.bytes+size>self.max_bytes:
                        _,(_,old_size,_) = self.entries.popitem(last=False)
                        self.bytes-=old_size
                    self.entries[keys[i]]=(value,size,now)
                    self.bytes+=size
        values=[resolved[key] for key in keys]
        for key in keys:
            if key in self.entries:self.entries.move_to_end(key)
        return values[0][2](pooler_output=tuple(v[0] for v in values),
                            deepstack_features=[torch.cat([v[1][layer] for v in values])
                                                for layer in range(len(values[0][1]))])
