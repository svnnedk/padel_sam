"""Player detection using YOLOv8."""
import numpy as np
from typing import List, Dict, Optional
from pathlib import Path


class PlayerDetector:
    """Detect players in padel court frames using YOLOv8.

    This detector uses a pre-trained YOLOv8 model to detect persons,
    with optional fine-tuning for padel-specific detection.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: float = 0.5,
        device: str = 'auto'
    ):
        """Initialize the player detector.

        Args:
            model_path: Path to custom YOLO model, or None to use yolov8n
            confidence_threshold: Minimum confidence for detections
            device: Device to run inference on ('auto', 'cuda', 'cpu')
        """
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.model = None
        self.model_path = model_path

    def load_model(self):
        """Lazy load the YOLO model."""
        if self.model is not None:
            return

        try:
            from ultralytics import YOLO

            if self.model_path and Path(self.model_path).exists():
                self.model = YOLO(self.model_path)
            else:
                # Use pretrained YOLOv8 nano for speed
                self.model = YOLO('yolov8n.pt')

            # Set device
            if self.device != 'auto':
                self.model.to(self.device)

        except ImportError:
            raise ImportError(
                "ultralytics package required. Install with: pip install ultralytics"
            )

    def detect(
        self,
        frame: np.ndarray,
        classes: Optional[List[int]] = None
    ) -> List[Dict]:
        """Detect players in a frame.

        Args:
            frame: BGR image array
            classes: List of class IDs to detect (default: [0] for 'person')

        Returns:
            List of detections, each containing:
                - bbox: (x1, y1, x2, y2)
                - confidence: float
                - label: str
                - class_id: int
        """
        self.load_model()

        if classes is None:
            classes = [0]  # Person class in COCO

        # Run inference
        results = self.model(
            frame,
            conf=self.confidence_threshold,
            classes=classes,
            verbose=False
        )

        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for i in range(len(boxes)):
                bbox = boxes.xyxy[i].cpu().numpy()
                confidence = float(boxes.conf[i].cpu().numpy())
                class_id = int(boxes.cls[i].cpu().numpy())

                # Get class name
                label = self.model.names[class_id]

                detections.append({
                    'bbox': bbox.tolist(),
                    'confidence': confidence,
                    'label': label,
                    'class_id': class_id
                })

        return detections

    def detect_batch(
        self,
        frames: List[np.ndarray],
        classes: Optional[List[int]] = None
    ) -> List[List[Dict]]:
        """Detect players in multiple frames (batched inference)."""
        self.load_model()

        if classes is None:
            classes = [0]

        results = self.model(
            frames,
            conf=self.confidence_threshold,
            classes=classes,
            verbose=False
        )

        all_detections = []
        for result in results:
            frame_detections = []
            boxes = result.boxes
            if boxes is not None:
                for i in range(len(boxes)):
                    bbox = boxes.xyxy[i].cpu().numpy()
                    confidence = float(boxes.conf[i].cpu().numpy())
                    class_id = int(boxes.cls[i].cpu().numpy())
                    label = self.model.names[class_id]

                    frame_detections.append({
                        'bbox': bbox.tolist(),
                        'confidence': confidence,
                        'label': label,
                        'class_id': class_id
                    })
            all_detections.append(frame_detections)

        return all_detections

    def classify_team(
        self,
        detections: List[Dict],
        frame_height: int
    ) -> List[Dict]:
        """Classify players into teams based on their position.

        In padel, teams are typically on opposite sides of the net.
        This uses a simple heuristic based on y-position.

        Args:
            detections: List of player detections
            frame_height: Height of the frame

        Returns:
            Detections with 'team' field added ('near' or 'far')
        """
        mid_y = frame_height / 2

        for det in detections:
            bbox = det['bbox']
            # Use bottom of bounding box (feet position)
            feet_y = bbox[3]

            # Near team is closer to camera (larger y values)
            det['team'] = 'near' if feet_y > mid_y else 'far'

        return detections

    def filter_court_area(
        self,
        detections: List[Dict],
        court_bounds: Optional[np.ndarray] = None,
        margin: float = 50
    ) -> List[Dict]:
        """Filter detections to only include players within court bounds.

        Args:
            detections: List of detections
            court_bounds: 4x2 array of court corner points
            margin: Margin in pixels around court

        Returns:
            Filtered detections within court area
        """
        if court_bounds is None:
            return detections

        import cv2

        # Create court polygon with margin
        bounds = court_bounds.copy()
        center = bounds.mean(axis=0)
        bounds = bounds + (bounds - center) * (margin / np.linalg.norm(bounds - center, axis=1, keepdims=True))

        filtered = []
        for det in detections:
            bbox = det['bbox']
            # Check if center of bbox is inside court
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2

            if cv2.pointPolygonTest(bounds.astype(np.float32), (center_x, center_y), False) >= 0:
                filtered.append(det)

        return filtered
