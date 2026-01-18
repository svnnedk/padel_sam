"""Ball detection for padel matches.

Ball detection is challenging due to:
- Small size of the ball
- High speed causing motion blur
- Similar color to court/walls
"""
import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
from collections import deque


class BallDetector:
    """Detect padel ball using multiple techniques.

    Combines:
    1. Color-based detection (yellow ball)
    2. Motion-based detection (frame differencing)
    3. ML-based detection (optional, using YOLO)
    """

    # Padel ball is typically yellow
    BALL_HSV_LOWER = np.array([20, 100, 100])
    BALL_HSV_UPPER = np.array([35, 255, 255])

    def __init__(
        self,
        method: str = 'hybrid',
        model_path: Optional[str] = None,
        min_radius: int = 5,
        max_radius: int = 30,
        history_length: int = 10
    ):
        """Initialize ball detector.

        Args:
            method: Detection method ('color', 'motion', 'ml', 'hybrid')
            model_path: Path to ball detection model (for 'ml' method)
            min_radius: Minimum ball radius in pixels
            max_radius: Maximum ball radius in pixels
            history_length: Number of frames to keep for motion analysis
        """
        self.method = method
        self.model_path = model_path
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.history_length = history_length

        self.frame_history = deque(maxlen=history_length)
        self.position_history = deque(maxlen=history_length)
        self.model = None

    def detect(self, frame: np.ndarray) -> Optional[Dict]:
        """Detect ball in frame.

        Args:
            frame: BGR image

        Returns:
            Detection dict with 'center', 'radius', 'confidence', 'bbox'
            or None if no ball detected
        """
        # Store frame for motion analysis
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.frame_history.append(gray)

        if self.method == 'color':
            detection = self._detect_by_color(frame)
        elif self.method == 'motion':
            detection = self._detect_by_motion(frame)
        elif self.method == 'ml':
            detection = self._detect_by_ml(frame)
        else:  # hybrid
            detection = self._detect_hybrid(frame)

        # Update position history
        if detection is not None:
            self.position_history.append(detection['center'])
        else:
            self.position_history.append(None)

        return detection

    def _detect_by_color(self, frame: np.ndarray) -> Optional[Dict]:
        """Detect ball by color segmentation."""
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Create mask for yellow ball
        mask = cv2.inRange(hsv, self.BALL_HSV_LOWER, self.BALL_HSV_UPPER)

        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return None

        # Find best ball candidate
        best_candidate = None
        best_score = 0

        for contour in contours:
            # Fit circle
            ((x, y), radius) = cv2.minEnclosingCircle(contour)

            if not (self.min_radius <= radius <= self.max_radius):
                continue

            # Calculate circularity
            area = cv2.contourArea(contour)
            if area == 0:
                continue
            circularity = 4 * np.pi * area / (cv2.arcLength(contour, True) ** 2)

            # Score based on circularity and size
            score = circularity * min(radius / self.min_radius, 1.0)

            if score > best_score:
                best_score = score
                best_candidate = {
                    'center': (int(x), int(y)),
                    'radius': int(radius),
                    'confidence': min(circularity, 1.0),
                    'bbox': [
                        int(x - radius),
                        int(y - radius),
                        int(x + radius),
                        int(y + radius)
                    ],
                    'label': 'ball'
                }

        return best_candidate

    def _detect_by_motion(self, frame: np.ndarray) -> Optional[Dict]:
        """Detect ball by motion (frame differencing)."""
        if len(self.frame_history) < 2:
            return None

        # Calculate frame difference
        prev_frame = self.frame_history[-2]
        curr_frame = self.frame_history[-1]

        diff = cv2.absdiff(prev_frame, curr_frame)

        # Threshold
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

        # Find moving objects
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return None

        # Find small, circular moving objects (likely ball)
        best_candidate = None
        best_score = 0

        for contour in contours:
            ((x, y), radius) = cv2.minEnclosingCircle(contour)

            # Ball should be small and moving
            if not (self.min_radius <= radius <= self.max_radius * 1.5):
                continue

            area = cv2.contourArea(contour)
            if area == 0:
                continue

            circularity = 4 * np.pi * area / (cv2.arcLength(contour, True) ** 2)

            # Predict based on previous positions
            predicted_pos = self._predict_position()
            if predicted_pos is not None:
                dist_to_predicted = np.sqrt(
                    (x - predicted_pos[0])**2 + (y - predicted_pos[1])**2
                )
                prediction_bonus = max(0, 1 - dist_to_predicted / 200)
            else:
                prediction_bonus = 0

            score = circularity + prediction_bonus * 0.5

            if score > best_score:
                best_score = score
                best_candidate = {
                    'center': (int(x), int(y)),
                    'radius': int(radius),
                    'confidence': min(score, 1.0),
                    'bbox': [
                        int(x - radius),
                        int(y - radius),
                        int(x + radius),
                        int(y + radius)
                    ],
                    'label': 'ball'
                }

        return best_candidate

    def _detect_by_ml(self, frame: np.ndarray) -> Optional[Dict]:
        """Detect ball using ML model."""
        if self.model is None:
            self._load_model()

        if self.model is None:
            return self._detect_by_color(frame)

        results = self.model(frame, verbose=False)

        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue

            # Get highest confidence ball detection
            for i in range(len(boxes)):
                bbox = boxes.xyxy[i].cpu().numpy()
                confidence = float(boxes.conf[i].cpu().numpy())

                x1, y1, x2, y2 = bbox
                center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                radius = int(max(x2 - x1, y2 - y1) / 2)

                return {
                    'center': center,
                    'radius': radius,
                    'confidence': confidence,
                    'bbox': bbox.tolist(),
                    'label': 'ball'
                }

        return None

    def _detect_hybrid(self, frame: np.ndarray) -> Optional[Dict]:
        """Combine multiple detection methods for robustness."""
        # Try color detection first (most reliable when ball is visible)
        color_det = self._detect_by_color(frame)

        # Try motion detection
        motion_det = self._detect_by_motion(frame)

        # Combine results
        if color_det is None and motion_det is None:
            return None

        if color_det is None:
            return motion_det

        if motion_det is None:
            return color_det

        # Both detected - check if they agree
        dist = np.sqrt(
            (color_det['center'][0] - motion_det['center'][0])**2 +
            (color_det['center'][1] - motion_det['center'][1])**2
        )

        if dist < 30:  # Same ball
            # Average the positions, boost confidence
            cx = (color_det['center'][0] + motion_det['center'][0]) // 2
            cy = (color_det['center'][1] + motion_det['center'][1]) // 2
            radius = (color_det['radius'] + motion_det['radius']) // 2
            confidence = min(
                color_det['confidence'] + motion_det['confidence'] * 0.5,
                1.0
            )

            return {
                'center': (cx, cy),
                'radius': radius,
                'confidence': confidence,
                'bbox': [cx - radius, cy - radius, cx + radius, cy + radius],
                'label': 'ball'
            }
        else:
            # Different detections - use color (more reliable)
            return color_det

    def _predict_position(self) -> Optional[Tuple[float, float]]:
        """Predict next ball position based on trajectory."""
        # Get recent valid positions
        valid_positions = [p for p in self.position_history if p is not None]

        if len(valid_positions) < 2:
            return None

        # Simple linear prediction
        p1 = valid_positions[-2]
        p2 = valid_positions[-1]

        vx = p2[0] - p1[0]
        vy = p2[1] - p1[1]

        return (p2[0] + vx, p2[1] + vy)

    def _load_model(self):
        """Load ML model for ball detection."""
        if self.model_path is None:
            return

        try:
            from ultralytics import YOLO
            self.model = YOLO(self.model_path)
        except Exception as e:
            print(f"Failed to load ball detection model: {e}")
            self.model = None

    def get_trajectory(self) -> List[Optional[Tuple[int, int]]]:
        """Get recent ball trajectory."""
        return list(self.position_history)

    def estimate_velocity(self) -> Optional[Tuple[float, float]]:
        """Estimate ball velocity in pixels per frame."""
        valid_positions = [p for p in self.position_history if p is not None]

        if len(valid_positions) < 2:
            return None

        # Calculate average velocity over recent frames
        velocities = []
        for i in range(1, len(valid_positions)):
            vx = valid_positions[i][0] - valid_positions[i-1][0]
            vy = valid_positions[i][1] - valid_positions[i-1][1]
            velocities.append((vx, vy))

        avg_vx = sum(v[0] for v in velocities) / len(velocities)
        avg_vy = sum(v[1] for v in velocities) / len(velocities)

        return (avg_vx, avg_vy)
