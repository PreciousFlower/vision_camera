"""CPU person detection and conservative anonymous association within one camera.

YOLOX decoding follows the OpenCV Zoo model's documented RGB/letterbox format.
Model source: https://github.com/opencv/opencv_zoo/tree/main/models/object_detection_yolox
No face recognition or inference of an employee's real identity.
"""
from __future__ import annotations
import math
import threading
import time
from pathlib import Path


class PersonDetector:
    def __init__(self, path: Path):
        import cv2
        import numpy as np
        if not path.is_file(): raise FileNotFoundError('人员模型未准备，请运行 tools/prepare_person_detector.py')
        try:
            import onnxruntime as ort
            options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
            self.session=ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider'])
            self.input_name=self.session.get_inputs()[0].name
        except ImportError:
            self.session=None
            self.net = cv2.dnn.readNetFromONNX(str(path))
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        grids, strides = [], []
        for step in (8, 16, 32):
            x, y = np.meshgrid(np.arange(640//step), np.arange(640//step))
            grids.append(np.stack((x, y), axis=-1).reshape(-1, 2))
            strides.append(np.full((len(grids[-1]), 1), step))
        self.grid, self.strides = np.concatenate(grids), np.concatenate(strides)
        self.lock = threading.Lock()
        self.last_latency = 0

    def detect(self, image):
        import cv2
        import numpy as np
        started = time.perf_counter()
        height, width = image.shape[:2]
        scale = min(640/width, 640/height)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        padded = np.full((640, 640, 3), 114, dtype=np.float32)
        resized = cv2.resize(rgb, (max(1, int(width*scale)), max(1, int(height*scale))))
        padded[:resized.shape[0], :resized.shape[1]] = resized
        with self.lock:
            tensor=padded.transpose(2,0,1)[None].copy()
            if self.session is not None: output=self.session.run(None,{self.input_name:tensor})[0].reshape(-1,85).copy()
            else:
                self.net.setInput(tensor)
                output = self.net.forward().reshape(-1, 85).copy()
        centers = (output[:, :2] + self.grid) * self.strides
        sizes = np.exp(np.clip(output[:, 2:4], -15, 15)) * self.strides
        scores = output[:, 4] * output[:, 5]
        # COCO class zero is person; reject another class with greater evidence.
        valid = np.flatnonzero((scores >= .35) & (np.argmax(output[:, 5:], axis=1) == 0))
        boxes = np.concatenate((centers-sizes/2, sizes), axis=1)[valid] / scale
        keep = cv2.dnn.NMSBoxes(boxes.tolist(), scores[valid].tolist(), .35, .45)
        detections = []
        for position in np.asarray(keep).reshape(-1):
            x, y, w, h = boxes[int(position)]
            box = [max(0., x/width), max(0., y/height), min(1., (x+w)/width), min(1., (y+h)/height)]
            if box[2] <= box[0] or box[3] <= box[1]: continue
            crop = image[int(box[1]*height):int(box[3]*height), int(box[0]*width):int(box[2]*width)]
            histogram = cv2.calcHist([cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)], [0, 1], None, [12, 8], [0, 180, 0, 256])
            cv2.normalize(histogram, histogram, norm_type=cv2.NORM_L1)
            detections.append({'bbox': [float(v) for v in box], 'confidence': float(scores[valid[int(position)]]),
                               '_appearance': histogram.reshape(-1).tolist()})
        self.last_latency = time.perf_counter()-started
        return detections[:30]


def overlap(a, b):
    area = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1])-area
    return area/union if union > 0 else 0


class AnonymousTracker:
    def __init__(self, max_age=3):
        self.max_age, self.tracks, self.sequence = max_age, {}, 0

    def update(self, timestamp, detections):
        candidates, choices = [], {}
        active = {pid: value for pid, value in self.tracks.items() if timestamp-value['seen_at'] <= self.max_age}
        for i, detection in enumerate(detections):
            box = detection['bbox']
            costs = []
            for pid, old in active.items():
                previous = old['bbox']
                dt = max(.001, timestamp-old['seen_at'])
                predicted = [previous[j]+old.get('velocity', [0]*4)[j]*min(dt, 1) for j in range(4)]
                distance = math.hypot((box[0]+box[2]-predicted[0]-predicted[2])/2,
                                      (box[1]+box[3]-predicted[1]-predicted[3])/2)
                appearance, prior = detection.get('_appearance'), old.get('_appearance')
                color = sum(abs(x-y) for x, y in zip(appearance, prior))/2 if appearance and prior else 0
                iou = overlap(box, predicted)
                if distance > .25 or (iou < .03 and distance > .12) or color > .6: continue
                cost = .55*(1-iou) + .8*distance + .45*color
                costs.append((cost, pid))
            costs.sort()
            ambiguous = len(costs)>1 and costs[1][0]-costs[0][0] < .08
            choices[i] = [pid for _, pid in costs[:2]] if ambiguous else []
            if not ambiguous:
                candidates.extend((cost, i, pid) for cost, pid in costs)
        assigned, used = {}, set()
        for cost, i, pid in sorted(candidates):
            if i not in assigned and pid not in used:
                assigned[i] = pid; used.add(pid)
        output = []
        for i, detection in enumerate(detections):
            pid = assigned.get(i)
            if pid is None:
                self.sequence += 1; pid = f'P{self.sequence:03d}'
            old = self.tracks.get(pid)
            velocity = [(detection['bbox'][j]-old['bbox'][j])/max(.001, timestamp-old['seen_at']) for j in range(4)] if old else [0]*4
            track = {**detection, 'person_id': pid, 'seen_at': timestamp, 'velocity': velocity,
                     'pending_links': choices[i], 'identity_status': 'pending' if choices[i] else 'anonymous'}
            self.tracks[pid] = track
            output.append({k: v for k, v in track.items() if not k.startswith('_') and k != 'velocity'})
        # Only a short association memory; old IDs remain in the time ledger.
        self.tracks = {pid: value for pid, value in self.tracks.items() if timestamp-value['seen_at'] <= self.max_age}
        return output
