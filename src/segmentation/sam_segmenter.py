"""Player segmentation using SAM2/SAM3.

SAM (Segment Anything Model) provides precise segmentation masks
that can be used for:
- Precise player boundary detection
- Player pose analysis
- Jersey color identification for team classification
- Occlusion handling
"""
import numpy as np
from typing import List, Dict, Optional, Tuple
from pathlib import Path


class SAMSegmenter:
    """Segment players using SAM2/SAM3 with bounding box prompts.

    SAM provides pixel-precise masks given detection bounding boxes,
    which is useful for detailed player analysis.
    """

    def __init__(
        self,
        model_type: str = 'sam2_hiera_tiny',
        device: str = 'auto',
        model_path: Optional[str] = None
    ):
        """Initialize SAM segmenter.

        Args:
            model_type: SAM model variant
                - 'sam2_hiera_tiny': Fastest, suitable for real-time
                - 'sam2_hiera_small': Good balance
                - 'sam2_hiera_base_plus': Better accuracy
                - 'sam2_hiera_large': Best accuracy
            device: 'auto', 'cuda', 'cpu'
            model_path: Path to model checkpoint (downloads if None)
        """
        self.model_type = model_type
        self.device = device
        self.model_path = model_path

        self.model = None
        self.predictor = None
        self._image_set = False
        self._current_image = None

    def load_model(self):
        """Lazy load SAM model."""
        if self.model is not None:
            return

        try:
            # Try SAM2 first (newer)
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            import torch

            # Determine device
            if self.device == 'auto':
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
            else:
                device = self.device

            # Model configs for SAM2
            model_configs = {
                'sam2_hiera_tiny': 'sam2_hiera_t.yaml',
                'sam2_hiera_small': 'sam2_hiera_s.yaml',
                'sam2_hiera_base_plus': 'sam2_hiera_b+.yaml',
                'sam2_hiera_large': 'sam2_hiera_l.yaml',
            }

            config = model_configs.get(self.model_type, 'sam2_hiera_t.yaml')

            # Build model
            if self.model_path:
                checkpoint = self.model_path
            else:
                # Use default checkpoint path
                checkpoint = f"checkpoints/{self.model_type}.pt"

            self.model = build_sam2(config, checkpoint, device=device)
            self.predictor = SAM2ImagePredictor(self.model)

            print(f"Loaded SAM2 model: {self.model_type} on {device}")

        except ImportError:
            # Fall back to original SAM
            try:
                from segment_anything import sam_model_registry, SamPredictor
                import torch

                if self.device == 'auto':
                    device = 'cuda' if torch.cuda.is_available() else 'cpu'
                else:
                    device = self.device

                # Original SAM model types
                sam_type_map = {
                    'sam2_hiera_tiny': 'vit_b',
                    'sam2_hiera_small': 'vit_b',
                    'sam2_hiera_base_plus': 'vit_l',
                    'sam2_hiera_large': 'vit_h',
                }
                sam_type = sam_type_map.get(self.model_type, 'vit_b')

                if self.model_path:
                    checkpoint = self.model_path
                else:
                    checkpoint = f"checkpoints/sam_{sam_type}.pth"

                self.model = sam_model_registry[sam_type](checkpoint=checkpoint)
                self.model.to(device)
                self.predictor = SamPredictor(self.model)

                print(f"Loaded SAM model: {sam_type} on {device}")

            except ImportError:
                raise ImportError(
                    "Neither SAM2 nor SAM is installed. "
                    "Install with: pip install segment-anything-2 or pip install segment-anything"
                )

    def set_image(self, image: np.ndarray):
        """Set image for segmentation.

        Call this once per frame, then segment multiple objects.

        Args:
            image: BGR image (will be converted to RGB)
        """
        self.load_model()

        # Convert BGR to RGB
        image_rgb = image[:, :, ::-1]

        self.predictor.set_image(image_rgb)
        self._image_set = True
        self._current_image = image

    def segment_box(
        self,
        bbox: Tuple[int, int, int, int],
        multimask: bool = False
    ) -> Dict:
        """Segment object within bounding box.

        Args:
            bbox: (x1, y1, x2, y2) bounding box
            multimask: If True, return multiple mask options

        Returns:
            Dict with:
                - 'mask': Binary mask (H, W)
                - 'score': Confidence score
                - 'masks': All masks if multimask=True
                - 'scores': All scores if multimask=True
        """
        if not self._image_set:
            raise RuntimeError("Call set_image() first")

        box = np.array(bbox)

        masks, scores, _ = self.predictor.predict(
            box=box,
            multimask_output=multimask
        )

        if multimask:
            # Return best mask and all options
            best_idx = np.argmax(scores)
            return {
                'mask': masks[best_idx],
                'score': float(scores[best_idx]),
                'masks': masks,
                'scores': scores.tolist()
            }
        else:
            return {
                'mask': masks[0],
                'score': float(scores[0])
            }

    def segment_point(
        self,
        point: Tuple[int, int],
        point_label: int = 1,
        multimask: bool = True
    ) -> Dict:
        """Segment object at point.

        Args:
            point: (x, y) point coordinates
            point_label: 1 for foreground, 0 for background
            multimask: If True, return multiple mask options

        Returns:
            Dict with mask and score
        """
        if not self._image_set:
            raise RuntimeError("Call set_image() first")

        point_coords = np.array([[point[0], point[1]]])
        point_labels = np.array([point_label])

        masks, scores, _ = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=multimask
        )

        if multimask:
            best_idx = np.argmax(scores)
            return {
                'mask': masks[best_idx],
                'score': float(scores[best_idx]),
                'masks': masks,
                'scores': scores.tolist()
            }
        else:
            return {
                'mask': masks[0],
                'score': float(scores[0])
            }

    def segment_detections(
        self,
        detections: List[Dict]
    ) -> List[Dict]:
        """Segment all detected objects.

        Args:
            detections: List of detection dicts with 'bbox'

        Returns:
            Detections with 'mask' and 'mask_score' added
        """
        if not self._image_set:
            raise RuntimeError("Call set_image() first")

        for det in detections:
            bbox = det['bbox']
            result = self.segment_box(tuple(map(int, bbox)))
            det['mask'] = result['mask']
            det['mask_score'] = result['score']

        return detections

    def extract_player_colors(
        self,
        mask: np.ndarray,
        image: Optional[np.ndarray] = None
    ) -> Dict:
        """Extract dominant colors from player region.

        Useful for team classification based on jersey colors.

        Args:
            mask: Binary mask of player
            image: BGR image (uses current if None)

        Returns:
            Dict with 'dominant_colors', 'histogram'
        """
        if image is None:
            image = self._current_image

        if image is None:
            raise RuntimeError("No image available")

        import cv2

        # Apply mask to image
        masked = cv2.bitwise_and(image, image, mask=mask.astype(np.uint8))

        # Convert to HSV for better color analysis
        hsv = cv2.cvtColor(masked, cv2.COLOR_BGR2HSV)

        # Get non-zero pixels
        pixels = hsv[mask > 0]

        if len(pixels) == 0:
            return {'dominant_colors': [], 'histogram': None}

        # Calculate color histogram
        hist_h = cv2.calcHist([pixels], [0], None, [30], [0, 180])
        hist_s = cv2.calcHist([pixels], [1], None, [32], [0, 256])

        # Find dominant hue
        dominant_hue = np.argmax(hist_h) * 6  # Scale back to 0-180

        # Classify color
        color_name = self._hue_to_color_name(dominant_hue)

        return {
            'dominant_colors': [color_name],
            'dominant_hue': dominant_hue,
            'histogram': {
                'hue': hist_h.flatten().tolist(),
                'saturation': hist_s.flatten().tolist()
            }
        }

    def _hue_to_color_name(self, hue: float) -> str:
        """Convert HSV hue to color name."""
        if hue < 10 or hue > 170:
            return 'red'
        elif hue < 25:
            return 'orange'
        elif hue < 35:
            return 'yellow'
        elif hue < 85:
            return 'green'
        elif hue < 130:
            return 'blue'
        elif hue < 160:
            return 'purple'
        else:
            return 'pink'

    def get_mask_area(self, mask: np.ndarray) -> int:
        """Get area of mask in pixels."""
        return int(np.sum(mask > 0))

    def get_mask_centroid(self, mask: np.ndarray) -> Tuple[int, int]:
        """Get centroid of mask."""
        import cv2
        moments = cv2.moments(mask.astype(np.uint8))
        if moments['m00'] == 0:
            return (0, 0)
        cx = int(moments['m10'] / moments['m00'])
        cy = int(moments['m01'] / moments['m00'])
        return (cx, cy)

    def get_mask_bbox(self, mask: np.ndarray) -> Tuple[int, int, int, int]:
        """Get bounding box of mask."""
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)

        if not np.any(rows) or not np.any(cols):
            return (0, 0, 0, 0)

        y1, y2 = np.where(rows)[0][[0, -1]]
        x1, x2 = np.where(cols)[0][[0, -1]]

        return (int(x1), int(y1), int(x2), int(y2))
