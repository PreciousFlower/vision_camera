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


def _clothing_appearance(crop):
    """Two central garment patches, stable on gray clothing and bin boundaries."""
    import cv2
    # Lab chroma avoids undefined hue on gray clothes. Smooth the histogram to
    # tolerate small webcam white-balance changes; retain garment brightness.
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    height, width = lab.shape[:2]
    descriptor = []
    for top, bottom in ((.30, .55), (.55, .80)):
        patch = lab[int(height*top):max(int(height*top)+1, int(height*bottom)),
                    int(width*.20):max(int(width*.20)+1, int(width*.80))]
        chroma = cv2.calcHist([patch], [1, 2], None, [16, 16], [0, 256, 0, 256])
        chroma = cv2.GaussianBlur(chroma, (5, 5), 1.15)
        cv2.normalize(chroma, chroma, norm_type=cv2.NORM_L1)
        brightness = cv2.calcHist([patch], [0], None, [8], [0, 256])
        brightness = cv2.GaussianBlur(brightness, (1, 3), .8)
        cv2.normalize(brightness, brightness, norm_type=cv2.NORM_L1)
        descriptor.extend((chroma.reshape(-1)*.35).tolist())
        descriptor.extend((brightness.reshape(-1)*.15).tolist())
    return descriptor


class PersonDetector:
    def __init__(self, path: Path, score_threshold=.5):
        import cv2
        import numpy as np
        if not isinstance(score_threshold, (int, float)) or isinstance(score_threshold, bool) or not .05 <= score_threshold <= .95:
            raise ValueError('人员检测置信度阈值必须在 0.05–0.95 之间')
        self.score_threshold = float(score_threshold)
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
        valid = np.flatnonzero((scores >= self.score_threshold) & (np.argmax(output[:, 5:], axis=1) == 0))
        boxes = np.concatenate((centers-sizes/2, sizes), axis=1)[valid] / scale
        keep = cv2.dnn.NMSBoxes(boxes.tolist(), scores[valid].tolist(), self.score_threshold, .45)
        detections = []
        for position in np.asarray(keep).reshape(-1):
            x, y, w, h = boxes[int(position)]
            box = [max(0., x/width), max(0., y/height), min(1., (x+w)/width), min(1., (y+h)/height)]
            if box[2] <= box[0] or box[3] <= box[1]: continue
            crop = image[int(box[1]*height):int(box[3]*height), int(box[0]*width):int(box[2]*width)]
            if not crop.size: continue
            # A full detection rectangle contains background, shoes and changing
            # arm poses. Use two central clothing patches so a return through a
            # different doorway can still match the same anonymous person.
            detections.append({'bbox': [float(v) for v in box], 'confidence': float(scores[valid[int(position)]]),
                               '_appearance': _clothing_appearance(crop)})
        self.last_latency = time.perf_counter()-started
        return detections[:30]


def overlap(a, b):
    area = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1])-area
    return area/union if union > 0 else 0


def _appearance_distance(left, right):
    """Histogram total variation; absent descriptors never prove a return."""
    if not left or not right or len(left) != len(right):
        return None
    if (not all(isinstance(v, (int, float)) and math.isfinite(v) and v >= 0 for v in (*left, *right))
            or sum(left) <= 0 or sum(right) <= 0):
        return None
    ltotal, rtotal = sum(left), sum(right)
    return sum(abs(a/ltotal-b/rtotal) for a, b in zip(left, right))/2


def _center_distance(left, right):
    return math.hypot((left[0]+left[2]-right[0]-right[2])/2,
                      (left[1]+left[3]-right[1]-right[3])/2)


class AnonymousTracker:
    """Short motion tracks plus bounded clothing memory for observable returns.

    A gap is not evidence that an employee left. Return association requires a
    unique strong clothing match; similar absent people remain pending. This is
    anonymous association for one uninterrupted session, not face recognition.
    """
    def __init__(self, max_age=3, gallery_age=8*60*60, gallery_limit=500):
        self.max_age, self.tracks, self.sequence = max_age, {}, 0
        self.gallery_age, self.gallery_limit = gallery_age, gallery_limit

    @staticmethod
    def _color(detection, old):
        distances = [_appearance_distance(detection.get('_appearance'), sample)
                     for sample in old.get('_appearance_samples', [old.get('_appearance')])]
        return min((v for v in distances if v is not None), default=None)

    @staticmethod
    def _exit_evidence(box, old, velocity):
        if old is None:
            return False
        previous = old['bbox']
        vx, vy = (velocity[0]+velocity[2])/2, (velocity[1]+velocity[3])/2
        # Moving toward an actual image boundary distinguishes a visible exit
        # from a stationary person hidden behind a desk or a failed detection.
        return bool((box[0] <= .04 and vx < -.025 and box[0] < previous[0])
                    or (box[2] >= .96 and vx > .025 and box[2] > previous[2])
                    or (box[1] <= .03 and vy < -.025 and box[1] < previous[1])
                    or (box[3] >= .97 and vy > .04 and box[3] > previous[3]))

    def update(self, timestamp, detections):
        self.tracks = {pid: old for pid, old in self.tracks.items()
                       if timestamp-old['seen_at'] <= self.gallery_age}
        candidates, choices = [], {i: [] for i in range(len(detections))}
        active = {pid: old for pid, old in self.tracks.items()
                  if timestamp-old['seen_at'] <= self.max_age}
        for i, detection in enumerate(detections):
            costs = []
            for pid, old in active.items():
                dt = max(.001, timestamp-old['seen_at'])
                predicted = [old['bbox'][j]+old.get('velocity', [0]*4)[j]*min(dt, .5) for j in range(4)]
                distance = _center_distance(detection['bbox'], predicted)
                color = self._color(detection, old)
                iou = overlap(detection['bbox'], predicted)
                if distance > .25 or (iou < .03 and distance > .12) or (color is not None and color > .6):
                    continue
                costs.append((.55*(1-iou)+.8*distance+.45*(color or 0), pid))
            costs.sort()
            ambiguous = len(costs) > 1 and costs[1][0]-costs[0][0] < .08
            if ambiguous:
                # Keep a newly created pending track stable on subsequent
                # frames; do not mint another pending ID for the same box.
                stable = [item for item in costs if active[item[1]].get('identity_status') == 'pending'
                          and overlap(detection['bbox'], active[item[1]]['bbox']) >= .65]
                if len(stable) == 1:
                    candidates.append((stable[0][0], i, stable[0][1]))
                else:
                    choices[i] = [pid for _, pid in costs[:2]]
            else:
                candidates.extend((cost, i, pid) for cost, pid in costs)
        assigned, used, methods = {}, set(), {}
        for cost, i, pid in sorted(candidates):
            if i not in assigned and pid not in used:
                assigned[i] = pid; used.add(pid); methods[i] = 'motion'

        # Appearance returns are independent of the previous image position.
        # Reserve all motion matches first so two current people cannot share an
        # identity. Both candidate and detection must have a unique best match.
        returns = {}
        for i, detection in enumerate(detections):
            if i in assigned or choices[i]:
                continue
            costs = []
            for pid, old in self.tracks.items():
                if pid in used or old.get('identity_status') == 'pending':
                    continue
                color = self._color(detection, old)
                if color is None or color > .28:
                    continue
                dt = timestamp-old['seen_at']
                if dt <= self.max_age and (_center_distance(detection['bbox'], old['bbox']) > min(.6, .18+max(0.,dt)*.25)
                                           or color > .18):
                    continue
                h = detection['bbox'][3]-detection['bbox'][1]
                prior_h = old['bbox'][3]-old['bbox'][1]
                if not .3 <= h/max(.001, prior_h) <= 3:
                    continue
                costs.append((color, pid))
            costs.sort()
            if len(costs) > 1 and costs[1][0]-costs[0][0] < .12:
                choices[i] = [pid for _, pid in costs[:5]]
            elif costs:
                returns[i] = costs[0]
        by_identity = {}
        for i, (cost, pid) in returns.items():
            by_identity.setdefault(pid, []).append((cost, i))
        for pid, options in by_identity.items():
            options.sort()
            if len(options) > 1 and options[1][0]-options[0][0] < .12:
                for _, i in options:
                    choices[i] = [pid]
                continue
            _, i = options[0]
            assigned[i] = pid; used.add(pid); methods[i] = 'appearance_return'

        output = []
        for i, detection in enumerate(detections):
            pid = assigned.get(i)
            if pid is None:
                self.sequence += 1; pid = f'P{self.sequence:03d}'
            old = self.tracks.get(pid)
            dt = max(.001, timestamp-old['seen_at']) if old else 1
            velocity = [(detection['bbox'][j]-old['bbox'][j])/dt for j in range(4)] if old and dt <= self.max_age else [0]*4
            samples = list(old.get('_appearance_samples', [])) if old else []
            appearance = detection.get('_appearance')
            if _appearance_distance(appearance, appearance) is not None:
                comparable = [distance for s in samples
                              if (distance := _appearance_distance(appearance,s)) is not None]
                if not comparable or min(comparable) > .035:
                    samples.append(list(appearance))
            pending = choices[i] or (old.get('pending_links', []) if old else [])
            track = {**detection, 'person_id': pid, 'seen_at': timestamp, 'velocity': velocity,
                     '_appearance_samples': samples[-8:], 'pending_links': pending,
                     'identity_status': 'pending' if pending else 'anonymous',
                     'identity_method': methods.get(i, 'new'),
                     'departure_evidence': self._exit_evidence(detection['bbox'], old, velocity)}
            self.tracks[pid] = track
            output.append({k: v for k, v in track.items() if not k.startswith('_') and k != 'velocity'})
        if len(self.tracks) > self.gallery_limit:
            # Current visible people are never evicted to make room for old
            # gallery entries; session ledger has its own explicit person cap.
            visible_ids = {p['person_id'] for p in output}
            retained = sorted(self.tracks, key=lambda pid: (pid in visible_ids, self.tracks[pid]['seen_at']), reverse=True)
            self.tracks = {pid:self.tracks[pid] for pid in retained[:max(self.gallery_limit, len(visible_ids))]}
        return output
