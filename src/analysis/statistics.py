"""Match statistics extraction and analysis."""
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass, field
import json


@dataclass
class PlayerStats:
    """Statistics for a single player."""
    player_id: int
    team: str = 'unknown'

    # Position stats
    positions: List[Tuple[float, float]] = field(default_factory=list)
    total_distance: float = 0.0  # meters
    time_in_zones: Dict[str, float] = field(default_factory=dict)

    # Movement stats
    avg_speed: float = 0.0  # m/s
    max_speed: float = 0.0
    sprints: int = 0  # moments above sprint threshold

    # Court coverage
    heatmap_data: List[Tuple[float, float]] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            'player_id': self.player_id,
            'team': self.team,
            'total_distance_m': round(self.total_distance, 2),
            'avg_speed_ms': round(self.avg_speed, 2),
            'max_speed_ms': round(self.max_speed, 2),
            'sprints': self.sprints,
            'time_in_zones': self.time_in_zones,
            'position_count': len(self.positions)
        }


@dataclass
class MatchStats:
    """Overall match statistics."""
    total_frames: int = 0
    total_duration: float = 0.0  # seconds
    fps: float = 30.0

    # Rally stats
    total_rallies: int = 0
    avg_rally_length: float = 0.0  # seconds
    longest_rally: float = 0.0
    rally_lengths: List[float] = field(default_factory=list)

    # Ball stats
    ball_detections: int = 0
    avg_ball_speed: float = 0.0  # estimated

    # Player stats
    player_stats: Dict[int, PlayerStats] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            'total_frames': self.total_frames,
            'total_duration_s': round(self.total_duration, 2),
            'fps': self.fps,
            'total_rallies': self.total_rallies,
            'avg_rally_length_s': round(self.avg_rally_length, 2),
            'longest_rally_s': round(self.longest_rally, 2),
            'ball_detections': self.ball_detections,
            'players': {
                pid: stats.to_dict()
                for pid, stats in self.player_stats.items()
            }
        }


class MatchStatistics:
    """Compute and aggregate match statistics."""

    # Court dimensions for distance calculation
    COURT_LENGTH_M = 20.0
    COURT_WIDTH_M = 10.0

    # Speed thresholds (m/s)
    SPRINT_THRESHOLD = 4.0  # ~14 km/h

    def __init__(self, fps: float = 30.0):
        """Initialize statistics tracker.

        Args:
            fps: Video frames per second
        """
        self.fps = fps
        self.stats = MatchStats(fps=fps)

        # Frame-by-frame data
        self.frame_data = []

        # Per-player tracking
        self._player_prev_positions = {}
        self._player_speeds = defaultdict(list)

    def update(
        self,
        frame_idx: int,
        player_detections: List[Dict],
        ball_detection: Optional[Dict] = None,
        rally_active: bool = False
    ):
        """Update statistics with new frame data.

        Args:
            frame_idx: Current frame index
            player_detections: List of player detections with court positions
            ball_detection: Ball detection if found
            rally_active: Whether a rally is currently in progress
        """
        self.stats.total_frames = max(self.stats.total_frames, frame_idx + 1)
        self.stats.total_duration = self.stats.total_frames / self.fps

        # Store frame data
        self.frame_data.append({
            'frame': frame_idx,
            'players': player_detections,
            'ball': ball_detection,
            'rally_active': rally_active
        })

        # Update player statistics
        for det in player_detections:
            track_id = det.get('track_id')
            if track_id is None:
                continue

            # Initialize player stats if needed
            if track_id not in self.stats.player_stats:
                self.stats.player_stats[track_id] = PlayerStats(
                    player_id=track_id,
                    team=det.get('team', 'unknown')
                )

            player = self.stats.player_stats[track_id]

            # Get court position
            court_pos = det.get('court_position')
            if court_pos is not None:
                player.positions.append(court_pos)
                player.heatmap_data.append(court_pos)

                # Calculate distance and speed
                if track_id in self._player_prev_positions:
                    prev_pos = self._player_prev_positions[track_id]
                    distance = self._calculate_distance(prev_pos, court_pos)
                    player.total_distance += distance

                    # Speed in m/s
                    speed = distance * self.fps
                    self._player_speeds[track_id].append(speed)

                    if speed > player.max_speed:
                        player.max_speed = speed

                    if speed > self.SPRINT_THRESHOLD:
                        player.sprints += 1

                self._player_prev_positions[track_id] = court_pos

        # Ball statistics
        if ball_detection is not None:
            self.stats.ball_detections += 1

    def _calculate_distance(
        self,
        pos1: Tuple[float, float],
        pos2: Tuple[float, float]
    ) -> float:
        """Calculate real-world distance between two court positions."""
        dx = (pos2[0] - pos1[0]) * self.COURT_WIDTH_M
        dy = (pos2[1] - pos1[1]) * self.COURT_LENGTH_M
        return np.sqrt(dx**2 + dy**2)

    def finalize(self):
        """Finalize statistics after processing all frames."""
        # Calculate average speeds
        for track_id, speeds in self._player_speeds.items():
            if track_id in self.stats.player_stats and speeds:
                self.stats.player_stats[track_id].avg_speed = np.mean(speeds)

        # Calculate time in zones for each player
        for track_id, player in self.stats.player_stats.items():
            zone_counts = defaultdict(int)
            for pos in player.positions:
                zone = self._classify_zone(pos)
                zone_counts[zone] += 1

            total = len(player.positions)
            if total > 0:
                player.time_in_zones = {
                    zone: count / total
                    for zone, count in zone_counts.items()
                }

    def _classify_zone(self, position: Tuple[float, float]) -> str:
        """Classify position into court zone."""
        x, y = position

        # Determine side
        side = 'left' if x < 0.5 else 'right'

        # Determine depth (relative to nearest baseline)
        y_rel = y if y < 0.5 else (1 - y)
        if y_rel < 0.2:
            depth = 'back'
        elif y_rel < 0.4:
            depth = 'mid'
        else:
            depth = 'net'

        return f"{depth}_{side}"

    def get_player_heatmap(
        self,
        player_id: int,
        resolution: Tuple[int, int] = (50, 100)
    ) -> np.ndarray:
        """Generate position heatmap for a player.

        Args:
            player_id: Player track ID
            resolution: (width, height) of heatmap grid

        Returns:
            2D numpy array with position density
        """
        if player_id not in self.stats.player_stats:
            return np.zeros(resolution[::-1])

        player = self.stats.player_stats[player_id]
        heatmap = np.zeros(resolution[::-1])

        for pos in player.heatmap_data:
            x = int(pos[0] * (resolution[0] - 1))
            y = int(pos[1] * (resolution[1] - 1))
            if 0 <= x < resolution[0] and 0 <= y < resolution[1]:
                heatmap[y, x] += 1

        return heatmap

    def get_team_heatmap(
        self,
        team: str,
        resolution: Tuple[int, int] = (50, 100)
    ) -> np.ndarray:
        """Generate combined heatmap for a team."""
        heatmap = np.zeros(resolution[::-1])

        for player in self.stats.player_stats.values():
            if player.team == team:
                player_heatmap = self.get_player_heatmap(
                    player.player_id, resolution
                )
                heatmap += player_heatmap

        return heatmap

    def export_stats(self, filepath: str):
        """Export statistics to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.stats.to_dict(), f, indent=2)

    def get_summary(self) -> str:
        """Get human-readable summary of statistics."""
        lines = [
            "=" * 50,
            "MATCH STATISTICS SUMMARY",
            "=" * 50,
            f"Duration: {self.stats.total_duration:.1f}s ({self.stats.total_frames} frames)",
            f"Ball detections: {self.stats.ball_detections}",
            "",
            "PLAYER STATISTICS:",
            "-" * 30,
        ]

        for track_id, player in self.stats.player_stats.items():
            lines.extend([
                f"Player {track_id} ({player.team} team):",
                f"  Distance covered: {player.total_distance:.1f}m",
                f"  Avg speed: {player.avg_speed:.2f} m/s ({player.avg_speed * 3.6:.1f} km/h)",
                f"  Max speed: {player.max_speed:.2f} m/s ({player.max_speed * 3.6:.1f} km/h)",
                f"  Sprints: {player.sprints}",
                ""
            ])

        return "\n".join(lines)
