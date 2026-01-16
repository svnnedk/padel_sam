"""Visualization utilities for padel analysis."""
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple


# Color palette for different players/teams
COLORS = {
    'team_near': (0, 255, 0),      # Green for near team
    'team_far': (0, 0, 255),       # Red for far team
    'ball': (0, 255, 255),          # Yellow for ball
    'court': (255, 255, 255),       # White for court lines
    'trajectory': (255, 165, 0),    # Orange for trajectories
}


class Visualizer:
    """Visualization tools for padel match analysis."""

    def __init__(self, court_width: int = 200, court_height: int = 400):
        """Initialize visualizer with mini-map dimensions."""
        self.court_width = court_width
        self.court_height = court_height

    def draw_detections(
        self,
        frame: np.ndarray,
        detections: List[Dict],
        show_labels: bool = True,
        show_masks: bool = True
    ) -> np.ndarray:
        """Draw bounding boxes and masks on frame."""
        result = frame.copy()

        for det in detections:
            bbox = det.get('bbox')
            label = det.get('label', '')
            confidence = det.get('confidence', 0)
            track_id = det.get('track_id')
            mask = det.get('mask')
            team = det.get('team', 'unknown')

            # Determine color based on team
            if 'ball' in label.lower():
                color = COLORS['ball']
            elif team == 'near':
                color = COLORS['team_near']
            else:
                color = COLORS['team_far']

            # Draw mask if available
            if show_masks and mask is not None:
                mask_colored = np.zeros_like(result)
                mask_colored[mask > 0] = color
                result = cv2.addWeighted(result, 1, mask_colored, 0.4, 0)

            # Draw bounding box
            if bbox is not None:
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)

                # Draw label
                if show_labels:
                    label_text = f"{label}"
                    if track_id is not None:
                        label_text = f"P{track_id}"
                    if confidence > 0:
                        label_text += f" {confidence:.2f}"

                    (w, h), _ = cv2.getTextSize(
                        label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                    )
                    cv2.rectangle(
                        result, (x1, y1 - h - 10), (x1 + w, y1), color, -1
                    )
                    cv2.putText(
                        result, label_text, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1
                    )

        return result

    def draw_court_lines(
        self,
        frame: np.ndarray,
        court_points: np.ndarray,
        color: Tuple[int, int, int] = (255, 255, 0)
    ) -> np.ndarray:
        """Draw detected court lines on frame."""
        result = frame.copy()
        if court_points is None or len(court_points) < 4:
            return result

        pts = court_points.astype(np.int32)
        cv2.polylines(result, [pts], True, color, 2)

        # Draw corner points
        for i, pt in enumerate(pts):
            cv2.circle(result, tuple(pt), 5, (0, 255, 0), -1)
            cv2.putText(
                result, str(i), tuple(pt + [5, 5]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
            )

        return result

    def draw_trajectory(
        self,
        frame: np.ndarray,
        positions: List[Tuple[int, int]],
        color: Tuple[int, int, int] = COLORS['trajectory'],
        max_points: int = 30
    ) -> np.ndarray:
        """Draw motion trajectory on frame."""
        result = frame.copy()
        positions = positions[-max_points:]

        for i in range(1, len(positions)):
            if positions[i - 1] is None or positions[i] is None:
                continue
            thickness = int(np.sqrt(max_points / float(i + 1)) * 2)
            cv2.line(result, positions[i - 1], positions[i], color, thickness)

        return result

    def create_minimap(
        self,
        player_positions: List[Dict],
        ball_position: Optional[Tuple[float, float]] = None
    ) -> np.ndarray:
        """Create a bird's eye view minimap of the court."""
        # Create court background
        minimap = np.zeros(
            (self.court_height, self.court_width, 3), dtype=np.uint8
        )
        minimap[:] = (50, 100, 50)  # Green court color

        # Draw court lines
        margin = 10
        cv2.rectangle(
            minimap,
            (margin, margin),
            (self.court_width - margin, self.court_height - margin),
            COLORS['court'], 2
        )
        # Service line
        cv2.line(
            minimap,
            (margin, self.court_height // 2),
            (self.court_width - margin, self.court_height // 2),
            COLORS['court'], 2
        )
        # Center line
        cv2.line(
            minimap,
            (self.court_width // 2, margin),
            (self.court_width // 2, self.court_height - margin),
            COLORS['court'], 1
        )

        # Draw players
        for player in player_positions:
            pos = player.get('court_position')
            if pos is None:
                continue

            x = int(pos[0] * (self.court_width - 2 * margin) + margin)
            y = int(pos[1] * (self.court_height - 2 * margin) + margin)

            team = player.get('team', 'unknown')
            color = COLORS['team_near'] if team == 'near' else COLORS['team_far']

            cv2.circle(minimap, (x, y), 8, color, -1)
            cv2.circle(minimap, (x, y), 8, (255, 255, 255), 1)

            track_id = player.get('track_id')
            if track_id is not None:
                cv2.putText(
                    minimap, str(track_id), (x - 3, y + 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1
                )

        # Draw ball
        if ball_position is not None:
            x = int(ball_position[0] * (self.court_width - 2 * margin) + margin)
            y = int(ball_position[1] * (self.court_height - 2 * margin) + margin)
            cv2.circle(minimap, (x, y), 5, COLORS['ball'], -1)

        return minimap

    def create_heatmap(
        self,
        positions: List[Tuple[float, float]],
        resolution: Tuple[int, int] = (100, 200)
    ) -> np.ndarray:
        """Create a position heatmap from normalized court coordinates."""
        heatmap = np.zeros(resolution, dtype=np.float32)

        for pos in positions:
            if pos is None:
                continue
            x = int(pos[0] * (resolution[0] - 1))
            y = int(pos[1] * (resolution[1] - 1))
            if 0 <= x < resolution[0] and 0 <= y < resolution[1]:
                heatmap[y, x] += 1

        # Apply Gaussian blur for smoothing
        heatmap = cv2.GaussianBlur(heatmap, (15, 15), 0)

        # Normalize and colorize
        if heatmap.max() > 0:
            heatmap = (heatmap / heatmap.max() * 255).astype(np.uint8)

        heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        return heatmap_colored

    def overlay_minimap(
        self,
        frame: np.ndarray,
        minimap: np.ndarray,
        position: str = 'bottom_right',
        scale: float = 1.0
    ) -> np.ndarray:
        """Overlay minimap on the main frame."""
        result = frame.copy()
        h, w = minimap.shape[:2]
        h, w = int(h * scale), int(w * scale)
        minimap_resized = cv2.resize(minimap, (w, h))

        margin = 10
        if position == 'bottom_right':
            y1, y2 = result.shape[0] - h - margin, result.shape[0] - margin
            x1, x2 = result.shape[1] - w - margin, result.shape[1] - margin
        elif position == 'top_right':
            y1, y2 = margin, h + margin
            x1, x2 = result.shape[1] - w - margin, result.shape[1] - margin
        elif position == 'bottom_left':
            y1, y2 = result.shape[0] - h - margin, result.shape[0] - margin
            x1, x2 = margin, w + margin
        else:  # top_left
            y1, y2 = margin, h + margin
            x1, x2 = margin, w + margin

        result[y1:y2, x1:x2] = minimap_resized

        return result
