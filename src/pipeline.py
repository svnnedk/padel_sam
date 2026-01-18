"""Main analysis pipeline for padel match videos."""
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List
from tqdm import tqdm

from .utils.video import VideoReader, VideoWriter
from .utils.visualization import Visualizer
from .court.detector import CourtDetector
from .court.homography import CourtHomography
from .detection.player_detector import PlayerDetector
from .detection.ball_detector import BallDetector
from .tracking.tracker import PlayerTracker
from .segmentation.sam_segmenter import SAMSegmenter
from .analysis.statistics import MatchStatistics
from .analysis.rally_detector import RallyDetector


class PadelAnalysisPipeline:
    """Complete pipeline for analyzing padel match videos.

    This pipeline combines:
    1. Court detection and homography transformation
    2. Player detection (YOLO) and tracking (ByteTrack)
    3. Precise player segmentation (SAM3)
    4. Ball detection and tracking
    5. Rally detection and statistics extraction
    """

    def __init__(
        self,
        court_corners: Optional[np.ndarray] = None,
        use_sam: bool = True,
        sam_model: str = 'sam2_hiera_tiny',
        device: str = 'auto',
        output_dir: str = 'outputs'
    ):
        """Initialize the analysis pipeline.

        Args:
            court_corners: Pre-defined court corners (4x2 array)
                          If None, will attempt auto-detection or prompt manual
            use_sam: Whether to use SAM for precise segmentation
            sam_model: SAM model variant to use
            device: Compute device ('auto', 'cuda', 'cpu')
            output_dir: Directory for output files
        """
        self.court_corners = court_corners
        self.use_sam = use_sam
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.court_detector = CourtDetector()
        self.homography = CourtHomography()
        self.player_detector = PlayerDetector(device=device)
        self.ball_detector = BallDetector(method='hybrid')
        self.tracker = PlayerTracker(tracker_type='bytetrack')
        self.visualizer = Visualizer()

        if use_sam:
            self.segmenter = SAMSegmenter(
                model_type=sam_model,
                device=device
            )
        else:
            self.segmenter = None

        self.statistics = None
        self.rally_detector = None

    def analyze_video(
        self,
        video_path: str,
        output_video: Optional[str] = None,
        frame_interval: int = 1,
        show_preview: bool = False,
        max_frames: Optional[int] = None
    ) -> Dict:
        """Analyze a padel match video.

        Args:
            video_path: Path to input video
            output_video: Path for annotated output video (optional)
            frame_interval: Process every Nth frame
            show_preview: Show live preview window
            max_frames: Maximum frames to process (for testing)

        Returns:
            Dictionary with analysis results
        """
        # Open video
        video = VideoReader(video_path)
        print(f"Video: {video.width}x{video.height} @ {video.fps:.2f} fps")
        print(f"Duration: {video.duration:.1f}s ({video.frame_count} frames)")

        # Initialize statistics
        self.statistics = MatchStatistics(fps=video.fps)
        self.rally_detector = RallyDetector(fps=video.fps)

        # Setup court homography
        self._setup_court(video)

        # Setup output video
        writer = None
        if output_video:
            writer = VideoWriter(
                output_video,
                fps=video.fps / frame_interval,
                width=video.width,
                height=video.height
            )

        # Process frames
        total_frames = min(video.frame_count, max_frames or float('inf'))
        pbar = tqdm(total=int(total_frames / frame_interval), desc="Processing")

        for frame_idx, frame in video.sample_frames(interval=frame_interval):
            if max_frames and frame_idx >= max_frames:
                break

            # Process frame
            result = self._process_frame(frame_idx, frame)

            # Visualize
            annotated = self._visualize_frame(frame, result)

            # Write output
            if writer:
                writer.write(annotated)

            # Show preview
            if show_preview:
                cv2.imshow('Padel Analysis', annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            pbar.update(1)

        pbar.close()

        if show_preview:
            cv2.destroyAllWindows()

        # Finalize statistics
        self.statistics.finalize()

        # Generate reports
        stats_path = self.output_dir / 'statistics.json'
        self.statistics.export_stats(str(stats_path))

        # Generate heatmaps
        self._generate_heatmaps()

        # Print summary
        print("\n" + self.statistics.get_summary())

        rally_stats = self.rally_detector.get_rally_stats()
        print(f"\nRally Statistics:")
        print(f"  Total rallies: {rally_stats['total_rallies']}")
        print(f"  Avg duration: {rally_stats['avg_duration']:.1f}s")
        print(f"  Avg shots: {rally_stats['avg_shots']:.1f}")

        return {
            'statistics': self.statistics.stats.to_dict(),
            'rally_stats': rally_stats,
            'output_video': output_video,
            'stats_file': str(stats_path)
        }

    def _setup_court(self, video: VideoReader):
        """Setup court detection and homography."""
        # Get first frame
        frame = video.get_frame(0)

        if self.court_corners is not None:
            # Use provided corners
            self.homography.compute_homography(self.court_corners)
        else:
            # Try auto-detection
            corners = self.court_detector.detect_court_corners(frame)

            if corners is not None:
                print("Court corners auto-detected")
                self.homography.compute_homography(corners)
                self.court_corners = corners
            else:
                print("Auto-detection failed. Using manual selection...")
                corners = self.court_detector.manual_court_corners(frame)
                if corners is not None:
                    self.homography.compute_homography(corners)
                    self.court_corners = corners
                else:
                    print("Warning: No court corners - position analysis disabled")

    def _process_frame(self, frame_idx: int, frame: np.ndarray) -> Dict:
        """Process a single frame.

        Returns dict with:
            - player_detections: List of player detections
            - ball_detection: Ball detection or None
            - rally_state: Current rally state
        """
        result = {
            'frame_idx': frame_idx,
            'player_detections': [],
            'ball_detection': None,
            'rally_state': None
        }

        # Detect players
        detections = self.player_detector.detect(frame)

        # Classify teams
        detections = self.player_detector.classify_team(
            detections, frame.shape[0]
        )

        # Track players
        detections = self.tracker.update(detections)

        # Add court positions
        if self.homography.is_valid():
            for det in detections:
                court_pos = self.homography.get_player_court_position(det['bbox'])
                det['court_position'] = court_pos

        # SAM segmentation (optional, for detailed analysis)
        if self.segmenter is not None and len(detections) > 0:
            self.segmenter.set_image(frame)
            detections = self.segmenter.segment_detections(detections)

        result['player_detections'] = detections

        # Detect ball
        ball = self.ball_detector.detect(frame)
        if ball is not None and self.homography.is_valid():
            ball_court_pos = self.homography.image_to_court(
                np.array([[ball['center'][0], ball['center'][1]]])
            )
            if ball_court_pos is not None:
                ball['court_position'] = tuple(ball_court_pos[0])

        result['ball_detection'] = ball

        # Update rally detection
        rally_state = self.rally_detector.update(
            frame_idx,
            ball,
            detections
        )
        result['rally_state'] = rally_state

        # Update statistics
        self.statistics.update(
            frame_idx,
            detections,
            ball,
            self.rally_detector.is_rally_active()
        )

        return result

    def _visualize_frame(self, frame: np.ndarray, result: Dict) -> np.ndarray:
        """Create annotated frame visualization."""
        annotated = frame.copy()

        # Draw player detections
        annotated = self.visualizer.draw_detections(
            annotated,
            result['player_detections'],
            show_masks=self.use_sam
        )

        # Draw ball
        if result['ball_detection'] is not None:
            ball = result['ball_detection']
            cv2.circle(
                annotated,
                ball['center'],
                ball['radius'] + 2,
                (0, 255, 255),
                2
            )

        # Draw ball trajectory
        trajectory = self.ball_detector.get_trajectory()
        trajectory = [p for p in trajectory if p is not None]
        if len(trajectory) > 1:
            annotated = self.visualizer.draw_trajectory(
                annotated,
                trajectory
            )

        # Draw court overlay if available
        if self.court_corners is not None:
            annotated = self.visualizer.draw_court_lines(
                annotated,
                self.court_corners
            )

        # Create and overlay minimap
        player_positions = [
            {
                'court_position': d.get('court_position'),
                'team': d.get('team'),
                'track_id': d.get('track_id')
            }
            for d in result['player_detections']
            if d.get('court_position') is not None
        ]

        ball_pos = None
        if result['ball_detection'] and 'court_position' in result['ball_detection']:
            ball_pos = result['ball_detection']['court_position']

        minimap = self.visualizer.create_minimap(player_positions, ball_pos)
        annotated = self.visualizer.overlay_minimap(annotated, minimap)

        # Add rally indicator
        if result['rally_state']:
            state_text = result['rally_state'].value.upper()
            color = (0, 255, 0) if 'rally' in state_text.lower() else (128, 128, 128)
            cv2.putText(
                annotated,
                f"Rally: {state_text}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                color,
                2
            )

        return annotated

    def _generate_heatmaps(self):
        """Generate and save player position heatmaps."""
        import matplotlib.pyplot as plt

        for track_id in self.statistics.stats.player_stats.keys():
            heatmap = self.statistics.get_player_heatmap(track_id)

            if heatmap.sum() > 0:
                plt.figure(figsize=(6, 12))
                plt.imshow(heatmap, cmap='hot', interpolation='gaussian')
                plt.title(f'Player {track_id} Position Heatmap')
                plt.colorbar(label='Time spent')
                plt.savefig(
                    self.output_dir / f'heatmap_player_{track_id}.png',
                    dpi=150,
                    bbox_inches='tight'
                )
                plt.close()

        # Team heatmaps
        for team in ['near', 'far']:
            heatmap = self.statistics.get_team_heatmap(team)
            if heatmap.sum() > 0:
                plt.figure(figsize=(6, 12))
                plt.imshow(heatmap, cmap='hot', interpolation='gaussian')
                plt.title(f'{team.capitalize()} Team Position Heatmap')
                plt.colorbar(label='Time spent')
                plt.savefig(
                    self.output_dir / f'heatmap_team_{team}.png',
                    dpi=150,
                    bbox_inches='tight'
                )
                plt.close()

    def analyze_frame(self, frame: np.ndarray) -> Dict:
        """Analyze a single frame (for real-time use).

        Args:
            frame: BGR image

        Returns:
            Analysis results for the frame
        """
        # Simplified single-frame analysis
        detections = self.player_detector.detect(frame)
        detections = self.player_detector.classify_team(detections, frame.shape[0])

        if self.segmenter is not None:
            self.segmenter.set_image(frame)
            detections = self.segmenter.segment_detections(detections)

        ball = self.ball_detector.detect(frame)

        return {
            'players': detections,
            'ball': ball
        }
