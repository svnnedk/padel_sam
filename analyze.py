#!/usr/bin/env python3
"""Main entry point for padel match analysis.

Usage:
    python analyze.py video.mp4 --output output.mp4
    python analyze.py video.mp4 --preview
    python analyze.py video.mp4 --no-sam --fast
"""
import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description='Analyze padel match videos using AI/ML',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full analysis with SAM segmentation
  python analyze.py match.mp4 --output analyzed.mp4

  # Quick analysis without SAM (faster)
  python analyze.py match.mp4 --no-sam --output analyzed.mp4

  # Preview mode (show live)
  python analyze.py match.mp4 --preview

  # Process only first 1000 frames (testing)
  python analyze.py match.mp4 --max-frames 1000

  # Use specific court corners (x1,y1,x2,y2,x3,y3,x4,y4)
  python analyze.py match.mp4 --court-corners 100,200,700,200,700,600,100,600
        """
    )

    parser.add_argument('video', help='Path to input video file')
    parser.add_argument('--output', '-o', help='Path for output video')
    parser.add_argument('--output-dir', default='outputs',
                        help='Directory for output files (default: outputs)')
    parser.add_argument('--preview', action='store_true',
                        help='Show live preview window')
    parser.add_argument('--no-sam', action='store_true',
                        help='Disable SAM segmentation (faster)')
    parser.add_argument('--sam-model', default='sam2_hiera_tiny',
                        choices=['sam2_hiera_tiny', 'sam2_hiera_small',
                                'sam2_hiera_base_plus', 'sam2_hiera_large'],
                        help='SAM model variant (default: sam2_hiera_tiny)')
    parser.add_argument('--device', default='auto',
                        choices=['auto', 'cuda', 'cpu'],
                        help='Compute device (default: auto)')
    parser.add_argument('--frame-interval', type=int, default=1,
                        help='Process every Nth frame (default: 1)')
    parser.add_argument('--max-frames', type=int,
                        help='Maximum frames to process (for testing)')
    parser.add_argument('--court-corners', type=str,
                        help='Court corners as comma-separated values: x1,y1,x2,y2,x3,y3,x4,y4')

    args = parser.parse_args()

    # Validate input
    if not Path(args.video).exists():
        print(f"Error: Video file not found: {args.video}")
        sys.exit(1)

    # Parse court corners if provided
    court_corners = None
    if args.court_corners:
        try:
            values = [float(v) for v in args.court_corners.split(',')]
            if len(values) != 8:
                raise ValueError("Need exactly 8 values for 4 corners")
            import numpy as np
            court_corners = np.array(values).reshape(4, 2)
        except Exception as e:
            print(f"Error parsing court corners: {e}")
            sys.exit(1)

    # Import here to avoid slow startup for --help
    from src.pipeline import PadelAnalysisPipeline

    # Create pipeline
    pipeline = PadelAnalysisPipeline(
        court_corners=court_corners,
        use_sam=not args.no_sam,
        sam_model=args.sam_model,
        device=args.device,
        output_dir=args.output_dir
    )

    # Run analysis
    results = pipeline.analyze_video(
        args.video,
        output_video=args.output,
        frame_interval=args.frame_interval,
        show_preview=args.preview,
        max_frames=args.max_frames
    )

    print("\n" + "=" * 50)
    print("Analysis complete!")
    print("=" * 50)

    if args.output:
        print(f"Output video: {args.output}")
    print(f"Statistics: {results['stats_file']}")
    print(f"Heatmaps: {args.output_dir}/heatmap_*.png")


if __name__ == '__main__':
    main()
