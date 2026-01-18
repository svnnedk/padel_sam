"""Homography transformation for court coordinate mapping."""
import cv2
import numpy as np
from typing import Optional, Tuple, List


class CourtHomography:
    """Transform between image coordinates and court coordinates.

    This allows mapping player positions from the camera view to a
    normalized top-down court view for analysis.
    """

    # Normalized court coordinates (0-1 range)
    # Top-down view: (0,0) = top-left, (1,1) = bottom-right
    COURT_CORNERS_NORMALIZED = np.array([
        [0.0, 0.0],  # top-left
        [1.0, 0.0],  # top-right
        [1.0, 1.0],  # bottom-right
        [0.0, 1.0],  # bottom-left
    ], dtype=np.float32)

    def __init__(self, image_corners: Optional[np.ndarray] = None):
        """Initialize with court corners in image coordinates."""
        self.image_corners = image_corners
        self.H = None  # Image to court homography
        self.H_inv = None  # Court to image homography

        if image_corners is not None:
            self.compute_homography(image_corners)

    def compute_homography(self, image_corners: np.ndarray):
        """Compute homography matrix from image corners.

        Args:
            image_corners: 4x2 array of corner points in image coordinates
                          ordered as [top-left, top-right, bottom-right, bottom-left]
        """
        self.image_corners = image_corners.astype(np.float32)

        # Compute homography: image -> normalized court
        self.H, _ = cv2.findHomography(
            self.image_corners,
            self.COURT_CORNERS_NORMALIZED
        )

        # Compute inverse: normalized court -> image
        self.H_inv, _ = cv2.findHomography(
            self.COURT_CORNERS_NORMALIZED,
            self.image_corners
        )

    def image_to_court(
        self,
        points: np.ndarray
    ) -> Optional[np.ndarray]:
        """Transform points from image coordinates to court coordinates.

        Args:
            points: Nx2 array of points in image coordinates

        Returns:
            Nx2 array of points in normalized court coordinates (0-1 range)
        """
        if self.H is None:
            return None

        points = np.array(points, dtype=np.float32)
        if points.ndim == 1:
            points = points.reshape(1, -1)

        # Apply homography
        points_homogeneous = np.hstack([
            points,
            np.ones((points.shape[0], 1))
        ])
        transformed = (self.H @ points_homogeneous.T).T

        # Convert from homogeneous coordinates
        transformed[:, 0] /= transformed[:, 2]
        transformed[:, 1] /= transformed[:, 2]

        return transformed[:, :2]

    def court_to_image(
        self,
        points: np.ndarray
    ) -> Optional[np.ndarray]:
        """Transform points from court coordinates to image coordinates.

        Args:
            points: Nx2 array of points in normalized court coordinates

        Returns:
            Nx2 array of points in image coordinates
        """
        if self.H_inv is None:
            return None

        points = np.array(points, dtype=np.float32)
        if points.ndim == 1:
            points = points.reshape(1, -1)

        # Apply inverse homography
        points_homogeneous = np.hstack([
            points,
            np.ones((points.shape[0], 1))
        ])
        transformed = (self.H_inv @ points_homogeneous.T).T

        # Convert from homogeneous coordinates
        transformed[:, 0] /= transformed[:, 2]
        transformed[:, 1] /= transformed[:, 2]

        return transformed[:, :2]

    def get_player_court_position(
        self,
        bbox: Tuple[int, int, int, int]
    ) -> Optional[Tuple[float, float]]:
        """Get player position on court from their bounding box.

        Uses the bottom-center of the bounding box as the player's foot position.

        Args:
            bbox: (x1, y1, x2, y2) bounding box

        Returns:
            (x, y) normalized court position
        """
        x1, y1, x2, y2 = bbox
        # Use bottom-center as foot position
        foot_x = (x1 + x2) / 2
        foot_y = y2

        court_pos = self.image_to_court(np.array([[foot_x, foot_y]]))
        if court_pos is not None:
            return tuple(court_pos[0])
        return None

    def create_topdown_view(
        self,
        frame: np.ndarray,
        output_size: Tuple[int, int] = (200, 400)
    ) -> np.ndarray:
        """Create a top-down view of the court.

        Args:
            frame: Input frame
            output_size: (width, height) of output

        Returns:
            Top-down warped view
        """
        if self.H is None:
            return np.zeros((*output_size[::-1], 3), dtype=np.uint8)

        # Scale normalized coords to output size
        scale_matrix = np.array([
            [output_size[0], 0, 0],
            [0, output_size[1], 0],
            [0, 0, 1]
        ], dtype=np.float32)

        H_scaled = scale_matrix @ self.H

        topdown = cv2.warpPerspective(
            frame,
            H_scaled,
            output_size
        )

        return topdown

    def is_valid(self) -> bool:
        """Check if homography has been computed."""
        return self.H is not None and self.H_inv is not None

    def estimate_distance(
        self,
        point1: Tuple[float, float],
        point2: Tuple[float, float],
        court_length_m: float = 20.0,
        court_width_m: float = 10.0
    ) -> float:
        """Estimate real-world distance between two court positions.

        Args:
            point1, point2: Normalized court coordinates
            court_length_m: Court length in meters
            court_width_m: Court width in meters

        Returns:
            Estimated distance in meters
        """
        dx = (point2[0] - point1[0]) * court_width_m
        dy = (point2[1] - point1[1]) * court_length_m
        return np.sqrt(dx**2 + dy**2)

    def classify_court_zone(
        self,
        position: Tuple[float, float]
    ) -> str:
        """Classify which zone of the court a position is in.

        Args:
            position: Normalized court coordinates (x, y)

        Returns:
            Zone name: 'net', 'service', 'back', 'left', 'right', etc.
        """
        x, y = position

        # Determine half (near=0-0.5, far=0.5-1)
        half = 'near' if y < 0.5 else 'far'

        # Determine side
        side = 'left' if x < 0.5 else 'right'

        # Determine depth
        y_rel = y if y < 0.5 else (1 - y)  # Distance from closest baseline
        if y_rel < 0.15:
            depth = 'back'
        elif y_rel < 0.35:
            depth = 'service'
        else:
            depth = 'net'

        return f"{half}_{depth}_{side}"
