import numpy as np
from collections import deque

use_lap = True
try:
    import lap
except ImportError:
    from scipy.optimize import linear_sum_assignment
    use_lap = False

class DotAccess(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

class TrackState:
    Active = 0
    Lost_Central = 1
    Lost_Marginal = 2

class Track:
    def __init__(self, bbox, frame_id, track_id, cls_id, score):
        self.track_id = track_id
        self.state = TrackState.Active
        self.class_history = deque(maxlen=10)
        self.score_history = deque(maxlen=10)
        self.update(bbox, frame_id, cls_id, score)

    def update(self, bbox, frame_id, cls_id, score):
        self.bbox = bbox
        self.cls_id = cls_id
        self.score = score
        self.state = TrackState.Active
        self.last_frame = frame_id
        self.class_history.append(cls_id)
        self.score_history.append(score)

    def get_smoothed_class(self):
        if not self.class_history:
            return self.cls_id
        weights = np.array(self.score_history)
        classes = np.array(self.class_history)
        return int(np.average(classes, weights=weights))

class SFSORT:
    def __init__(self, args):
        self.update_args(args)
        self.frame_no = 0
        self.id_counter = 0
        self.active_tracks = []
        self.lost_tracks = []
        self.confidence_history = deque(maxlen=100)

    def update_args(self, args):
        args = DotAccess(args)
        self.base_low_th = args.low_th
        self.base_match_th_second = args.match_th_second
        self.base_high_th = args.high_th
        self.base_match_th_first = args.match_th_first
        self.base_new_track_th = args.new_track_th
        self.marginal_timeout = args.marginal_timeout
        self.central_timeout = args.central_timeout
        self.l_margin = args.horizontal_margin
        self.t_margin = args.vertical_margin
        self.r_margin = args.frame_width - args.horizontal_margin
        self.b_margin = args.frame_height - args.vertical_margin
        self.frame_area = args.frame_width * args.frame_height

    def adaptive_thresholds(self, avg_confidence):
        factor = avg_confidence * 0.9
        self.low_th = self.base_low_th * factor
        self.match_th_second = self.base_match_th_second * factor
        self.high_th = self.base_high_th * factor
        self.match_th_first = self.base_match_th_first * factor
        self.new_track_th = self.base_new_track_th * factor

    def update(self, boxes, scores, class_ids):
        self.frame_no += 1
        
        # Update confidence history and calculate average
        self.confidence_history.extend(scores)
        avg_confidence = np.mean(self.confidence_history)
        
        # Adapt thresholds
        self.adaptive_thresholds(avg_confidence)

        # Filter small objects
        valid_indices = self.filter_small_objects(boxes)
        boxes = boxes[valid_indices]
        scores = scores[valid_indices]
        class_ids = class_ids[valid_indices]

        next_active_tracks = []

        # Remove long-time lost tracks
        self.remove_lost_tracks()

        # Gather all previous tracks
        track_pool = self.active_tracks + self.lost_tracks

        # Try to associate tracks with high score detections
        unmatched_tracks = np.array([])
        high_score = scores > self.high_th
        if high_score.any():
            definite_boxes = boxes[high_score]
            definite_scores = scores[high_score]
            definite_classes = class_ids[high_score]
            if track_pool:
                cost = self.calculate_cost(track_pool, definite_boxes)
                matches, unmatched_tracks, unmatched_detections = self.linear_assignment(cost, self.match_th_first)
                # Update/Activate matched tracks
                for track_idx, detection_idx in matches:
                    self.update_track(track_pool[track_idx], definite_boxes[detection_idx], 
                                      definite_classes[detection_idx], definite_scores[detection_idx])
                    next_active_tracks.append(track_pool[track_idx])
                    if track_pool[track_idx] in self.lost_tracks:
                        self.lost_tracks.remove(track_pool[track_idx])
                # Identify eligible unmatched detections as new tracks
                for detection_idx in unmatched_detections:
                    if definite_scores[detection_idx] > self.new_track_th:
                        self.create_new_track(definite_boxes[detection_idx], definite_classes[detection_idx], 
                                              definite_scores[detection_idx], next_active_tracks)
            else:
                # Associate tracks of the first frame after object-free/null frames
                for detection_idx, score in enumerate(definite_scores):
                    if score > self.new_track_th:
                        self.create_new_track(definite_boxes[detection_idx], definite_classes[detection_idx], 
                                              definite_scores[detection_idx], next_active_tracks)

        # Add unmatched tracks to the lost list
        unmatched_track_pool = [track_pool[i] for i in unmatched_tracks]
        next_lost_tracks = unmatched_track_pool.copy()

        # Try to associate remained tracks with intermediate score detections
        intermediate_score = np.logical_and((self.low_th < scores), (scores < self.high_th))
        if intermediate_score.any() and len(unmatched_tracks):
            possible_boxes = boxes[intermediate_score]
            possible_class_ids = class_ids[intermediate_score]
            possible_scores = scores[intermediate_score]
            cost = self.calculate_cost(unmatched_track_pool, possible_boxes, iou_only=True)
            matches, unmatched_tracks, unmatched_detections = self.linear_assignment(cost, self.match_th_second)
            # Update/Activate matched tracks
            for track_idx, detection_idx in matches:
                self.update_track(unmatched_track_pool[track_idx], possible_boxes[detection_idx], 
                                  possible_class_ids[detection_idx], possible_scores[detection_idx])
                next_active_tracks.append(unmatched_track_pool[track_idx])
                if unmatched_track_pool[track_idx] in self.lost_tracks:
                    self.lost_tracks.remove(unmatched_track_pool[track_idx])
                next_lost_tracks.remove(unmatched_track_pool[track_idx])

        # All tracks are lost if there are no detections
        if not (high_score.any() or intermediate_score.any()):
            next_lost_tracks = track_pool.copy()

        # Update the list of lost tracks
        self.update_lost_tracks(next_lost_tracks)

        # Update the list of active tracks
        self.active_tracks = next_active_tracks.copy()

        result = np.asarray([
            [x.bbox, x.track_id, x.get_smoothed_class(), round(x.score, 2)]
            for x in next_active_tracks],
            dtype=object)

        return result

    def filter_small_objects(self, boxes):
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        return areas > (0.001 * self.frame_area)

    def remove_lost_tracks(self):
        self.lost_tracks = [track for track in self.lost_tracks if not 
                            ((track.state == TrackState.Lost_Central and self.frame_no - track.last_frame > self.central_timeout) or
                             (track.state == TrackState.Lost_Marginal and self.frame_no - track.last_frame > self.marginal_timeout))]

    def update_track(self, track, box, cls_id, score):
        track.update(box, self.frame_no, cls_id, score)

    def create_new_track(self, box, cls_id, score, track_list):
        track = Track(box, self.frame_no, self.id_counter, cls_id, score)
        track_list.append(track)
        self.id_counter += 1

    def update_lost_tracks(self, next_lost_tracks):
        for track in next_lost_tracks:
            if track not in self.lost_tracks:
                self.lost_tracks.append(track)
                u = track.bbox[0] + (track.bbox[2] - track.bbox[0]/2)
                v = track.bbox[1] + (track.bbox[3] - track.bbox[1]/2)
                if (self.l_margin < u < self.r_margin) and (self.t_margin < v < self.b_margin):
                    track.state = TrackState.Lost_Central
                else:
                    track.state = TrackState.Lost_Marginal

    @staticmethod
    def calculate_cost(tracks, boxes, iou_only=False):
        eps = 1e-7
        active_boxes = [track.bbox for track in tracks]

        b1_x1, b1_y1, b1_x2, b1_y2 = np.array(active_boxes).T
        b2_x1, b2_y1, b2_x2, b2_y2 = np.array(boxes).T

        h_intersection = (np.minimum(b1_x2[:, None], b2_x2) - np.maximum(b1_x1[:, None], b2_x1)).clip(0)
        w_intersection = (np.minimum(b1_y2[:, None], b2_y2) - np.maximum(b1_y1[:, None], b2_y1)).clip(0)

        intersection =  h_intersection * w_intersection

        box1_height = b1_x2 - b1_x1
        box2_height = b2_x2 - b2_x1
        box1_width = b1_y2 - b1_y1
        box2_width = b2_y2 - b2_y1

        box1_area = box1_height * box1_width
        box2_area = box2_height * box2_width

        union = (box2_area + box1_area[:, None] - intersection + eps)

        iou = intersection / union

        if iou_only:
            return 1.0 - iou

        centerx1 = (b1_x1 + b1_x2) / 2.0
        centery1 = (b1_y1 + b1_y2) / 2.0
        centerx2 = (b2_x1 + b2_x2) / 2.0
        centery2 = (b2_y1 + b2_y2) / 2.0
        inner_diag = np.abs(centerx1[:, None] - centerx2) + np.abs(centery1[:, None] - centery2)

        xxc1 = np.minimum(b1_x1[:, None], b2_x1)
        yyc1 = np.minimum(b1_y1[:, None], b2_y1)
        xxc2 = np.maximum(b1_x2[:, None], b2_x2)
        yyc2 = np.maximum(b1_y2[:, None], b2_y2)
        outer_diag = np.abs(xxc2 - xxc1) + np.abs(yyc2 - yyc1)

        diou = iou - (inner_diag / outer_diag)

        delta_w = np.abs(box2_width - box1_width[:, None])
        sw = w_intersection / np.abs(w_intersection + delta_w + eps)

        delta_h = np.abs(box2_height - box1_height[:, None])
        sh = h_intersection / np.abs(h_intersection + delta_h + eps)

        bbsi = diou + sh + sw

        cost = (bbsi)/3.0

        return 1.0 - cost

    @staticmethod
    def linear_assignment(cost_matrix, thresh):
        if cost_matrix.size == 0:
            return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))

        if use_lap:
            _, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
            matches = [[ix, mx] for ix, mx in enumerate(x) if mx >= 0]
            unmatched_a = np.where(x < 0)[0]
            unmatched_b = np.where(y < 0)[0]
        else:
            y, x = linear_sum_assignment(cost_matrix)
            matches = np.asarray([[i, x] for i, x in enumerate(x) if cost_matrix[i, x] <= thresh])
            unmatched = np.ones(cost_matrix.shape)
            for i, xi in matches:
                unmatched[i, xi] = 0.0
            unmatched_a = np.where(unmatched.all(1))[0]
            unmatched_b = np.where(unmatched.all(0))[0]

        return matches, unmatched_a, unmatched_b
