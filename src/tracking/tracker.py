"""Multi-object tracking for players."""
import numpy as np
from typing import List, Dict, Optional
from collections import defaultdict


class PlayerTracker:
    """Track players across frames using ByteTrack or simple IoU tracking.

    Maintains consistent IDs for each player throughout the match.
    """

    def __init__(
        self,
        tracker_type: str = 'bytetrack',
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3
    ):
        """Initialize tracker.

        Args:
            tracker_type: 'bytetrack' or 'simple'
            max_age: Maximum frames to keep track without detection
            min_hits: Minimum detections before track is confirmed
            iou_threshold: IoU threshold for matching
        """
        self.tracker_type = tracker_type
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold

        self.tracks = {}
        self.next_id = 1
        self.frame_count = 0
        self.history = defaultdict(list)

        self._tracker = None

    def _init_bytetrack(self):
        """Initialize ByteTrack from supervision."""
        try:
            import supervision as sv
            self._tracker = sv.ByteTrack(
                track_activation_threshold=0.25,
                lost_track_buffer=self.max_age,
                minimum_matching_threshold=self.iou_threshold,
                frame_rate=30
            )
        except ImportError:
            print("supervision not available, falling back to simple tracker")
            self.tracker_type = 'simple'

    def update(self, detections: List[Dict]) -> List[Dict]:
        """Update tracks with new detections.

        Args:
            detections: List of detection dicts with 'bbox', 'confidence', etc.

        Returns:
            Detections with 'track_id' added
        """
        self.frame_count += 1

        if len(detections) == 0:
            self._age_tracks()
            return []

        if self.tracker_type == 'bytetrack':
            return self._update_bytetrack(detections)
        else:
            return self._update_simple(detections)

    def _update_bytetrack(self, detections: List[Dict]) -> List[Dict]:
        """Update using ByteTrack."""
        if self._tracker is None:
            self._init_bytetrack()

        if self._tracker is None:
            return self._update_simple(detections)

        import supervision as sv

        # Convert to supervision format
        xyxy = np.array([d['bbox'] for d in detections])
        confidence = np.array([d.get('confidence', 1.0) for d in detections])
        class_id = np.array([d.get('class_id', 0) for d in detections])

        sv_detections = sv.Detections(
            xyxy=xyxy,
            confidence=confidence,
            class_id=class_id
        )

        # Update tracker
        tracked = self._tracker.update_with_detections(sv_detections)

        # Add track IDs to detections
        tracked_detections = []
        for i, (track_id, det) in enumerate(zip(tracked.tracker_id, detections)):
            det['track_id'] = int(track_id) if track_id is not None else None
            tracked_detections.append(det)

            # Store in history
            if det['track_id'] is not None:
                self.history[det['track_id']].append({
                    'frame': self.frame_count,
                    'bbox': det['bbox'],
                    'team': det.get('team')
                })

        return tracked_detections

    def _update_simple(self, detections: List[Dict]) -> List[Dict]:
        """Simple IoU-based tracking."""
        if not self.tracks:
            # First frame - assign new IDs
            for det in detections:
                det['track_id'] = self.next_id
                self.tracks[self.next_id] = {
                    'bbox': det['bbox'],
                    'age': 0,
                    'hits': 1
                }
                self.history[self.next_id].append({
                    'frame': self.frame_count,
                    'bbox': det['bbox'],
                    'team': det.get('team')
                })
                self.next_id += 1
            return detections

        # Calculate IoU matrix
        track_ids = list(self.tracks.keys())
        track_boxes = [self.tracks[tid]['bbox'] for tid in track_ids]
        det_boxes = [d['bbox'] for d in detections]

        iou_matrix = self._compute_iou_matrix(track_boxes, det_boxes)

        # Greedy matching
        matched_tracks = set()
        matched_dets = set()

        # Sort by IoU and match greedily
        while True:
            max_iou = 0
            best_match = None

            for i, tid in enumerate(track_ids):
                if tid in matched_tracks:
                    continue
                for j in range(len(detections)):
                    if j in matched_dets:
                        continue
                    if iou_matrix[i, j] > max_iou:
                        max_iou = iou_matrix[i, j]
                        best_match = (tid, j)

            if best_match is None or max_iou < self.iou_threshold:
                break

            tid, det_idx = best_match
            matched_tracks.add(tid)
            matched_dets.add(det_idx)

            detections[det_idx]['track_id'] = tid
            self.tracks[tid]['bbox'] = detections[det_idx]['bbox']
            self.tracks[tid]['age'] = 0
            self.tracks[tid]['hits'] += 1

            self.history[tid].append({
                'frame': self.frame_count,
                'bbox': detections[det_idx]['bbox'],
                'team': detections[det_idx].get('team')
            })

        # Create new tracks for unmatched detections
        for j, det in enumerate(detections):
            if j not in matched_dets:
                det['track_id'] = self.next_id
                self.tracks[self.next_id] = {
                    'bbox': det['bbox'],
                    'age': 0,
                    'hits': 1
                }
                self.history[self.next_id].append({
                    'frame': self.frame_count,
                    'bbox': det['bbox'],
                    'team': det.get('team')
                })
                self.next_id += 1

        # Age unmatched tracks
        for tid in track_ids:
            if tid not in matched_tracks:
                self.tracks[tid]['age'] += 1
                if self.tracks[tid]['age'] > self.max_age:
                    del self.tracks[tid]

        return detections

    def _age_tracks(self):
        """Age all tracks when no detections."""
        for tid in list(self.tracks.keys()):
            self.tracks[tid]['age'] += 1
            if self.tracks[tid]['age'] > self.max_age:
                del self.tracks[tid]

    def _compute_iou_matrix(
        self,
        boxes1: List[List[float]],
        boxes2: List[List[float]]
    ) -> np.ndarray:
        """Compute IoU matrix between two sets of boxes."""
        n, m = len(boxes1), len(boxes2)
        iou_matrix = np.zeros((n, m))

        for i, box1 in enumerate(boxes1):
            for j, box2 in enumerate(boxes2):
                iou_matrix[i, j] = self._compute_iou(box1, box2)

        return iou_matrix

    def _compute_iou(
        self,
        box1: List[float],
        box2: List[float]
    ) -> float:
        """Compute IoU between two boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        if x2 < x1 or y2 < y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def get_track_history(self, track_id: int) -> List[Dict]:
        """Get history for a specific track."""
        return self.history.get(track_id, [])

    def get_active_tracks(self) -> Dict[int, Dict]:
        """Get currently active tracks."""
        return {
            tid: data for tid, data in self.tracks.items()
            if data['hits'] >= self.min_hits
        }

    def reset(self):
        """Reset tracker state."""
        self.tracks = {}
        self.next_id = 1
        self.frame_count = 0
        self.history = defaultdict(list)
        if self._tracker is not None:
            self._tracker.reset()
