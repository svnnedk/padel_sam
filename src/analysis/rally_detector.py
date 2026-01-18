"""Rally detection and scoring for padel matches."""
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class RallyState(Enum):
    """State of current rally."""
    IDLE = "idle"  # Between points
    SERVE = "serve"  # Serve in progress
    RALLY = "rally"  # Active rally
    POINT_ENDED = "point_ended"  # Point just ended


@dataclass
class Rally:
    """Data for a single rally/point."""
    start_frame: int
    end_frame: Optional[int] = None
    duration_frames: int = 0
    shot_count: int = 0
    winner: Optional[str] = None  # 'near' or 'far' team
    end_reason: Optional[str] = None  # 'winner', 'error', 'out', etc.

    @property
    def duration_seconds(self) -> float:
        return self.duration_frames / 30.0  # Assuming 30 fps


class RallyDetector:
    """Detect rallies and track scoring in padel matches.

    Uses ball tracking and player movement to determine:
    - When rallies start (serve)
    - When rallies end (point won/lost)
    - Shot counting
    """

    def __init__(
        self,
        fps: float = 30.0,
        min_rally_frames: int = 30,  # ~1 second minimum
        idle_threshold_frames: int = 60  # ~2 seconds without ball = point ended
    ):
        """Initialize rally detector.

        Args:
            fps: Video frames per second
            min_rally_frames: Minimum frames for valid rally
            idle_threshold_frames: Frames without ball to end rally
        """
        self.fps = fps
        self.min_rally_frames = min_rally_frames
        self.idle_threshold_frames = idle_threshold_frames

        self.state = RallyState.IDLE
        self.current_rally: Optional[Rally] = None
        self.rallies: List[Rally] = []

        # Tracking state
        self._frames_without_ball = 0
        self._last_ball_position = None
        self._ball_side_history = []  # Track which side ball is on

    def update(
        self,
        frame_idx: int,
        ball_detection: Optional[Dict],
        player_detections: List[Dict],
        court_y_mid: float = 0.5
    ) -> RallyState:
        """Update rally state with new frame data.

        Args:
            frame_idx: Current frame index
            ball_detection: Ball detection with 'court_position' if available
            player_detections: Player detections
            court_y_mid: Y coordinate of net (for side detection)

        Returns:
            Current rally state
        """
        ball_detected = ball_detection is not None
        ball_position = None

        if ball_detected:
            ball_position = ball_detection.get('court_position')
            self._frames_without_ball = 0
            self._last_ball_position = ball_position

            # Track which side the ball is on
            if ball_position is not None:
                side = 'near' if ball_position[1] > court_y_mid else 'far'
                self._ball_side_history.append(side)
        else:
            self._frames_without_ball += 1

        # State machine
        if self.state == RallyState.IDLE:
            # Look for serve (ball appears and moves)
            if ball_detected and len(self._ball_side_history) >= 3:
                self.state = RallyState.SERVE
                self.current_rally = Rally(start_frame=frame_idx)

        elif self.state == RallyState.SERVE:
            # Serve transitions to rally when ball crosses net
            if len(self._ball_side_history) >= 5:
                # Check if ball has crossed sides
                recent = self._ball_side_history[-5:]
                if len(set(recent)) > 1:  # Ball crossed net
                    self.state = RallyState.RALLY
                    if self.current_rally:
                        self.current_rally.shot_count = 1

            # Timeout on serve
            if self.current_rally and frame_idx - self.current_rally.start_frame > 150:
                self._end_rally(frame_idx, 'serve_timeout')

        elif self.state == RallyState.RALLY:
            if self.current_rally:
                self.current_rally.duration_frames = (
                    frame_idx - self.current_rally.start_frame
                )

                # Count shots (ball side changes)
                if len(self._ball_side_history) >= 2:
                    if self._ball_side_history[-1] != self._ball_side_history[-2]:
                        self.current_rally.shot_count += 1

            # Check for rally end
            if self._frames_without_ball > self.idle_threshold_frames:
                # Determine winner based on last ball position
                winner = self._determine_winner()
                self._end_rally(frame_idx, 'point_ended', winner)

        elif self.state == RallyState.POINT_ENDED:
            # Brief pause then return to idle
            if self._frames_without_ball > self.idle_threshold_frames * 2:
                self.state = RallyState.IDLE
                self._ball_side_history = []

        return self.state

    def _end_rally(
        self,
        frame_idx: int,
        reason: str,
        winner: Optional[str] = None
    ):
        """End current rally and record it."""
        if self.current_rally is None:
            return

        self.current_rally.end_frame = frame_idx
        self.current_rally.end_reason = reason
        self.current_rally.winner = winner

        # Only record if rally was long enough
        if self.current_rally.duration_frames >= self.min_rally_frames:
            self.rallies.append(self.current_rally)

        self.current_rally = None
        self.state = RallyState.POINT_ENDED

    def _determine_winner(self) -> Optional[str]:
        """Determine which team won the point.

        This is a heuristic based on where the ball was last seen.
        """
        if not self._ball_side_history:
            return None

        # If ball was last on 'near' side, 'far' team likely won
        # (ball not returned or went out on near side)
        last_side = self._ball_side_history[-1]
        return 'far' if last_side == 'near' else 'near'

    def is_rally_active(self) -> bool:
        """Check if a rally is currently in progress."""
        return self.state in (RallyState.SERVE, RallyState.RALLY)

    def get_rally_stats(self) -> Dict:
        """Get rally statistics."""
        if not self.rallies:
            return {
                'total_rallies': 0,
                'avg_duration': 0,
                'avg_shots': 0,
                'longest_rally': 0,
                'most_shots': 0
            }

        durations = [r.duration_seconds for r in self.rallies]
        shots = [r.shot_count for r in self.rallies]

        return {
            'total_rallies': len(self.rallies),
            'avg_duration': np.mean(durations),
            'avg_shots': np.mean(shots),
            'longest_rally': max(durations),
            'most_shots': max(shots),
            'near_team_points': sum(1 for r in self.rallies if r.winner == 'near'),
            'far_team_points': sum(1 for r in self.rallies if r.winner == 'far')
        }

    def get_rally_durations(self) -> List[float]:
        """Get list of rally durations in seconds."""
        return [r.duration_seconds for r in self.rallies]

    def reset(self):
        """Reset detector state."""
        self.state = RallyState.IDLE
        self.current_rally = None
        self.rallies = []
        self._frames_without_ball = 0
        self._last_ball_position = None
        self._ball_side_history = []
