"""Court line detection for padel courts."""
import cv2
import numpy as np
from typing import Optional, Tuple, List


class CourtDetector:
    """Detect padel court lines and keypoints.

    Padel court dimensions (meters):
    - Length: 20m (10m each side)
    - Width: 10m
    - Service box: 3m from net
    """

    # Standard padel court dimensions in meters
    COURT_LENGTH = 20.0
    COURT_WIDTH = 10.0
    SERVICE_LINE_DIST = 6.95  # From back wall

    def __init__(self, use_cached: bool = True):
        self.use_cached = use_cached
        self._cached_corners = None
        self._cache_frame_hash = None

    def detect_court_corners(
        self,
        frame: np.ndarray,
        method: str = 'hough'
    ) -> Optional[np.ndarray]:
        """Detect the four corners of the court.

        Returns corners in order: [top-left, top-right, bottom-right, bottom-left]
        from the camera's perspective.
        """
        if method == 'hough':
            return self._detect_via_hough(frame)
        elif method == 'color':
            return self._detect_via_color(frame)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _detect_via_hough(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Detect court lines using Hough transform."""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Edge detection
        edges = cv2.Canny(blurred, 50, 150)

        # Detect lines using Hough transform
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=100,
            minLineLength=100,
            maxLineGap=50
        )

        if lines is None:
            return None

        # Filter and cluster lines
        horizontal_lines = []
        vertical_lines = []

        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)

            if angle < 30 or angle > 150:  # Horizontal-ish
                horizontal_lines.append(line[0])
            elif 60 < angle < 120:  # Vertical-ish
                vertical_lines.append(line[0])

        # Find court boundaries from clustered lines
        corners = self._find_corners_from_lines(
            horizontal_lines, vertical_lines, frame.shape
        )

        return corners

    def _detect_via_color(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Detect court by color segmentation (blue court, white lines)."""
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Blue court mask (typical padel court)
        lower_blue = np.array([90, 50, 50])
        upper_blue = np.array([130, 255, 255])
        court_mask = cv2.inRange(hsv, lower_blue, upper_blue)

        # White lines mask
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 30, 255])
        lines_mask = cv2.inRange(hsv, lower_white, upper_white)

        # Find contours of the court
        contours, _ = cv2.findContours(
            court_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return None

        # Get largest contour (should be the court)
        largest_contour = max(contours, key=cv2.contourArea)

        # Approximate to polygon
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)

        if len(approx) >= 4:
            # Find the 4 corners using convex hull
            hull = cv2.convexHull(approx)
            corners = self._get_four_corners(hull)
            return corners

        return None

    def _find_corners_from_lines(
        self,
        h_lines: List,
        v_lines: List,
        frame_shape: Tuple[int, int]
    ) -> Optional[np.ndarray]:
        """Find corner points from detected lines."""
        if len(h_lines) < 2 or len(v_lines) < 2:
            return None

        h, w = frame_shape[:2]

        # Sort horizontal lines by y-coordinate (top to bottom)
        h_lines_sorted = sorted(h_lines, key=lambda l: (l[1] + l[3]) / 2)

        # Sort vertical lines by x-coordinate (left to right)
        v_lines_sorted = sorted(v_lines, key=lambda l: (l[0] + l[2]) / 2)

        # Get top and bottom horizontal lines
        top_line = h_lines_sorted[0]
        bottom_line = h_lines_sorted[-1]

        # Get left and right vertical lines
        left_line = v_lines_sorted[0]
        right_line = v_lines_sorted[-1]

        # Calculate intersections
        corners = []
        for h_line in [top_line, bottom_line]:
            for v_line in [left_line, right_line]:
                intersection = self._line_intersection(h_line, v_line)
                if intersection is not None:
                    corners.append(intersection)

        if len(corners) != 4:
            return None

        # Order corners: top-left, top-right, bottom-right, bottom-left
        corners = np.array(corners)
        corners = self._order_corners(corners)

        return corners

    def _line_intersection(
        self,
        line1: np.ndarray,
        line2: np.ndarray
    ) -> Optional[Tuple[float, float]]:
        """Calculate intersection point of two lines."""
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2

        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-10:
            return None

        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom

        px = x1 + t * (x2 - x1)
        py = y1 + t * (y2 - y1)

        return (px, py)

    def _get_four_corners(self, hull: np.ndarray) -> np.ndarray:
        """Extract four corners from a convex hull."""
        hull = hull.reshape(-1, 2)

        # Find the four extreme points
        top_left = hull[np.argmin(hull[:, 0] + hull[:, 1])]
        top_right = hull[np.argmax(hull[:, 0] - hull[:, 1])]
        bottom_right = hull[np.argmax(hull[:, 0] + hull[:, 1])]
        bottom_left = hull[np.argmax(-hull[:, 0] + hull[:, 1])]

        return np.array([top_left, top_right, bottom_right, bottom_left])

    def _order_corners(self, corners: np.ndarray) -> np.ndarray:
        """Order corners as: top-left, top-right, bottom-right, bottom-left."""
        # Sort by y first
        sorted_by_y = corners[np.argsort(corners[:, 1])]

        # Top two points
        top_two = sorted_by_y[:2]
        top_two = top_two[np.argsort(top_two[:, 0])]

        # Bottom two points
        bottom_two = sorted_by_y[2:]
        bottom_two = bottom_two[np.argsort(bottom_two[:, 0])]

        return np.array([
            top_two[0],      # top-left
            top_two[1],      # top-right
            bottom_two[1],   # bottom-right
            bottom_two[0]    # bottom-left
        ])

    def manual_court_corners(
        self,
        frame: np.ndarray,
        window_name: str = "Select Court Corners"
    ) -> np.ndarray:
        """Allow manual selection of court corners.

        User clicks on corners in order:
        top-left, top-right, bottom-right, bottom-left
        """
        corners = []
        display_frame = frame.copy()

        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN and len(corners) < 4:
                corners.append([x, y])
                cv2.circle(display_frame, (x, y), 5, (0, 255, 0), -1)
                cv2.putText(
                    display_frame, str(len(corners)),
                    (x + 10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2
                )
                if len(corners) > 1:
                    cv2.line(
                        display_frame,
                        tuple(corners[-2]), tuple(corners[-1]),
                        (0, 255, 0), 2
                    )
                cv2.imshow(window_name, display_frame)

        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, mouse_callback)

        print("Click on court corners in order:")
        print("1. Top-left  2. Top-right  3. Bottom-right  4. Bottom-left")
        print("Press 'r' to reset, 'q' to quit")

        cv2.imshow(window_name, display_frame)

        while len(corners) < 4:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('r'):
                corners.clear()
                display_frame = frame.copy()
                cv2.imshow(window_name, display_frame)
            elif key == ord('q'):
                break

        cv2.destroyWindow(window_name)

        if len(corners) == 4:
            return np.array(corners, dtype=np.float32)
        return None
