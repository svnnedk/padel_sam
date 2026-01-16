"""Video reading and writing utilities."""
import cv2
import numpy as np
from pathlib import Path
from typing import Iterator, Optional, Tuple


class VideoReader:
    """Efficient video reader with frame extraction capabilities."""

    def __init__(self, video_path: str):
        self.video_path = Path(video_path)
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        self.cap = cv2.VideoCapture(str(self.video_path))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.duration = self.frame_count / self.fps if self.fps > 0 else 0

    def __iter__(self) -> Iterator[Tuple[int, np.ndarray]]:
        """Iterate through all frames."""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_idx = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            yield frame_idx, frame
            frame_idx += 1

    def get_frame(self, frame_idx: int) -> Optional[np.ndarray]:
        """Get a specific frame by index."""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        return frame if ret else None

    def sample_frames(self, interval: int = 1) -> Iterator[Tuple[int, np.ndarray]]:
        """Sample frames at regular intervals."""
        for frame_idx, frame in self:
            if frame_idx % interval == 0:
                yield frame_idx, frame

    def __del__(self):
        if hasattr(self, 'cap'):
            self.cap.release()

    def __len__(self) -> int:
        return self.frame_count


class VideoWriter:
    """Video writer for saving processed frames."""

    def __init__(
        self,
        output_path: str,
        fps: float,
        width: int,
        height: int,
        codec: str = 'mp4v'
    ):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*codec)
        self.writer = cv2.VideoWriter(
            str(self.output_path), fourcc, fps, (width, height)
        )
        self.fps = fps
        self.width = width
        self.height = height

    def write(self, frame: np.ndarray):
        """Write a frame to the video."""
        if frame.shape[:2] != (self.height, self.width):
            frame = cv2.resize(frame, (self.width, self.height))
        self.writer.write(frame)

    def __del__(self):
        if hasattr(self, 'writer'):
            self.writer.release()
