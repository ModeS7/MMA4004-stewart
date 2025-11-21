#!/usr/bin/env python3
"""
Test Trained CNN Model on Recorded Video

Two-stage detection pipeline:
1. HSV color filtering → ROI extraction (~1ms)
2. CNN inference on crop → precise center (~2ms)

Shows comprehensive visualization of each stage.
"""

import cv2
import numpy as np
import time
import sys
from pathlib import Path

# Add parent directory to path for imports
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from ball_detection.detector import BallDetector


def process_frame(detector, frame, crop_size, confidence_threshold):
    """
    Process a single frame through the detection pipeline.

    Returns:
        result: (x, y, confidence) or None
        crop_vis: Visualization of CNN crop
        hsv_vis: Visualization of HSV mask
    """
    # Stage 1: ROI extraction
    crop, roi_center, crop_offset = detector.roi_extractor.extract_roi(frame)

    # Create HSV mask visualization
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, detector.roi_extractor.lower_red1, detector.roi_extractor.upper_red1)
    mask2 = cv2.inRange(hsv, detector.roi_extractor.lower_red2, detector.roi_extractor.upper_red2)
    hsv_mask = cv2.bitwise_or(mask1, mask2)
    hsv_mask = cv2.morphologyEx(hsv_mask, cv2.MORPH_OPEN, detector.roi_extractor.kernel)
    hsv_mask = cv2.morphologyEx(hsv_mask, cv2.MORPH_CLOSE, detector.roi_extractor.kernel)
    hsv_vis = cv2.cvtColor(hsv_mask, cv2.COLOR_GRAY2BGR)

    # Stage 2: CNN inference
    result = None
    crop_vis = np.zeros((crop_size, crop_size, 3), dtype=np.uint8)

    if crop is not None:
        x_norm, y_norm, confidence = detector.cnn_detector.detect(crop)

        if confidence >= confidence_threshold:
            # Convert to absolute coordinates
            x_offset, y_offset = crop_offset
            x_abs = x_offset + x_norm * crop_size
            y_abs = y_offset + y_norm * crop_size
            result = (x_abs, y_abs, confidence)

        # Visualize crop with CNN prediction
        crop_vis = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)

        # Draw CNN prediction on crop
        cx = int(x_norm * crop_size)
        cy = int(y_norm * crop_size)
        cv2.circle(crop_vis, (cx, cy), 3, (0, 255, 0), -1)
        cv2.circle(crop_vis, (cx, cy), 8, (0, 255, 0), 2)

        # Add crosshair
        cv2.line(crop_vis, (cx - 10, cy), (cx + 10, cy), (0, 255, 0), 1)
        cv2.line(crop_vis, (cx, cy - 10), (cx, cy + 10), (0, 255, 0), 1)

        # Add position info
        cv2.putText(crop_vis, f"Pos: ({x_norm:.2f}, {y_norm:.2f})",
                    (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        # Draw ROI box on HSV visualization
        if crop_offset is not None:
            x_off, y_off = crop_offset
            cv2.rectangle(hsv_vis, (x_off, y_off),
                          (x_off + crop_size, y_off + crop_size),
                          (0, 255, 0), 2)

    return result, crop_vis, hsv_vis


def test_video(video_path, model_path, use_gpu=False, crop_size=128, confidence_threshold=0.5):
    """
    Test ball detector on recorded stereo video with comprehensive visualization.

    Args:
        video_path: Path to stereo video file (side-by-side cameras)
        model_path: Path to trained ONNX model
        use_gpu: Whether to use DirectML GPU acceleration
        crop_size: Size of crop for CNN (must match training)
        confidence_threshold: Minimum confidence to accept detection
    """
    print("=" * 70)
    print("STEREO BALL DETECTION: HSV -> CNN")
    print("=" * 70)
    print(f"\nVideo: {video_path}")
    print(f"Model: {model_path}")
    print(f"Crop size: {crop_size}x{crop_size}")
    print(f"Backend: {'GPU (DirectML)' if use_gpu else 'CPU'}")
    print(f"Confidence threshold: {confidence_threshold}")

    # Check if files exist
    if not Path(video_path).exists():
        print(f"\nError: Video file not found: {video_path}")
        return

    if not Path(model_path).exists():
        print(f"\nError: Model file not found: {model_path}")
        return

    # Initialize detector
    print("\nInitializing detector...")
    detector = BallDetector(
        onnx_model_path=model_path,
        use_gpu=use_gpu,
        crop_size=crop_size,
        confidence_threshold=confidence_threshold
    )

    # Open video
    print(f"\nOpening video...")
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video file")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"\nVideo info:")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps:.1f}")
    print(f"  Total frames: {frame_count}")
    print(f"  Duration: {frame_count/fps:.1f}s")

    print("\nControls:")
    print("  q: Quit")
    print("  Space: Pause/Resume")
    print("  s: Print statistics")
    print("  r: Reset statistics")
    print("\nStarting playback...\n")

    # Performance tracking
    frame_idx = 0
    detections_left = 0
    detections_right = 0
    detections_stereo_pairs = 0
    detection_times = []
    roi_times = []
    cnn_times = []

    paused = False
    fps_start = time.time()
    fps_frames = 0

    try:
        while True:
            if not paused:
                ret, frame = cap.read()

                if not ret:
                    print("\nEnd of video")
                    break

                frame_idx += 1

                # Split stereo frame (2560x720 -> 1280x720 each)
                frame_left = frame[:, :width//2]
                frame_right = frame[:, width//2:]

                # ==== PROCESS LEFT CAMERA ====
                roi_start = time.time()
                result_left, crop_vis_left, hsv_vis_left = process_frame(
                    detector, frame_left, crop_size, confidence_threshold
                )
                roi_time_left = (time.time() - roi_start) * 1000

                # ==== PROCESS RIGHT CAMERA ====
                roi_start = time.time()
                result_right, crop_vis_right, hsv_vis_right = process_frame(
                    detector, frame_right, crop_size, confidence_threshold
                )
                roi_time_right = (time.time() - roi_start) * 1000

                # Track statistics
                total_time = roi_time_left + roi_time_right
                roi_times.append(total_time)

                if result_left:
                    detections_left += 1
                if result_right:
                    detections_right += 1
                if result_left and result_right:
                    detections_stereo_pairs += 1

                # ==== VISUALIZATION ====
                vis_left = detector.visualize(frame_left, result_left)
                vis_right = detector.visualize(frame_right, result_right)

                # Calculate FPS
                fps_frames += 1
                if fps_frames % 30 == 0:
                    current_fps = 30 / (time.time() - fps_start)
                    fps_start = time.time()
                else:
                    current_fps = fps if fps > 0 else 30

                # Add stats overlay to LEFT camera
                stats_y = 30
                cv2.putText(vis_left, f"LEFT CAMERA", (10, stats_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                stats_y += 35
                cv2.putText(vis_left, f"Frame: {frame_idx}/{frame_count}",
                            (10, stats_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                stats_y += 30
                cv2.putText(vis_left, f"FPS: {current_fps:.1f}",
                            (10, stats_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                stats_y += 30
                cv2.putText(vis_left, f"Left: {detections_left}/{frame_idx} ({detections_left/max(frame_idx,1)*100:.1f}%)",
                            (10, stats_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                stats_y += 25
                cv2.putText(vis_left, f"Right: {detections_right}/{frame_idx} ({detections_right/max(frame_idx,1)*100:.1f}%)",
                            (10, stats_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                stats_y += 25
                cv2.putText(vis_left, f"Stereo Pairs: {detections_stereo_pairs}",
                            (10, stats_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                stats_y += 25
                cv2.putText(vis_left, f"Time/Frame: {total_time:.1f}ms",
                            (10, stats_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

                # Detection status on LEFT
                cam_width = frame_left.shape[1]
                if result_left:
                    cv2.putText(vis_left, "DETECTED", (cam_width - 150, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    cv2.putText(vis_left, "NO DETECTION", (cam_width - 180, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # Add stats overlay to RIGHT camera
                cv2.putText(vis_right, f"RIGHT CAMERA", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                if result_right:
                    cv2.putText(vis_right, "DETECTED", (cam_width - 150, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    cv2.putText(vis_right, "NO DETECTION", (cam_width - 180, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # Add stage labels to HSV masks
                cv2.putText(hsv_vis_left, "STAGE 1: HSV Mask (Left)", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(hsv_vis_right, "STAGE 1: HSV Mask (Right)", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                # Add stage labels to CNN crops
                cv2.putText(crop_vis_left, "STAGE 2: CNN (Left)", (5, crop_size - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                cv2.putText(crop_vis_right, "STAGE 2: CNN (Right)", (5, crop_size - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

                # Combine visualizations (stack left and right horizontally)
                vis_combined = np.hstack([vis_left, vis_right])
                hsv_combined = np.hstack([hsv_vis_left, hsv_vis_right])
                crop_combined = np.hstack([crop_vis_left, crop_vis_right])

                # Display windows
                cv2.imshow('Detection Result', vis_combined)
                cv2.imshow('Stage 1: HSV Mask', hsv_combined)

                # Resize crop for better visibility
                crop_display = cv2.resize(crop_combined, (512, 256), interpolation=cv2.INTER_NEAREST)
                cv2.imshow('Stage 2: CNN Crop', crop_display)

            # Handle keyboard
            key = cv2.waitKey(1 if not paused else 0) & 0xFF

            if key == ord('q'):
                break
            elif key == ord(' '):
                paused = not paused
                print(f"{'Paused' if paused else 'Resumed'}")
            elif key == ord('s'):
                print_statistics(frame_idx, detections_left, detections_right,
                                detections_stereo_pairs, roi_times)
            elif key == ord('r'):
                frame_idx = 0
                detections_left = 0
                detections_right = 0
                detections_stereo_pairs = 0
                roi_times.clear()
                fps_frames = 0
                fps_start = time.time()
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                print("Statistics reset, video restarted")

    except KeyboardInterrupt:
        print("\nInterrupted by user")

    finally:
        cap.release()
        cv2.destroyAllWindows()

        # Final statistics
        print("\n" + "=" * 70)
        print("FINAL STATISTICS")
        print("=" * 70)
        print_statistics(frame_idx, detections_left, detections_right,
                        detections_stereo_pairs, roi_times)


def print_statistics(frame_count, detections_left, detections_right,
                     detections_stereo_pairs, roi_times):
    """Print detailed statistics."""
    print(f"\nFrames processed: {frame_count}")
    print(f"\nDetections:")
    print(f"  Left camera:  {detections_left} ({detections_left/max(frame_count,1)*100:.1f}%)")
    print(f"  Right camera: {detections_right} ({detections_right/max(frame_count,1)*100:.1f}%)")
    print(f"  Stereo pairs: {detections_stereo_pairs} ({detections_stereo_pairs/max(frame_count,1)*100:.1f}%)")

    if roi_times:
        print(f"\nPipeline time (both cameras):")
        print(f"  Mean: {np.mean(roi_times):.2f} ± {np.std(roi_times):.2f} ms")
        print(f"  Min:  {np.min(roi_times):.2f} ms")
        print(f"  Max:  {np.max(roi_times):.2f} ms")
        print(f"  P95:  {np.percentile(roi_times, 95):.2f} ms")
        print(f"  Mean per camera: {np.mean(roi_times)/2:.2f} ms")
        print(f"  Throughput: {1000/np.mean(roi_times):.1f} FPS")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Test Ball Detection on Video')
    parser.add_argument('--video', type=str,
                        default='ball_detection/ball_training_20251120_104443.mp4',
                        help='Path to video file')
    parser.add_argument('--model', type=str,
                        default='ball_detection/models/best_pixel_error.onnx',
                        help='Path to ONNX model')
    parser.add_argument('--gpu', action='store_true',
                        help='Use GPU acceleration (default: CPU)')
    parser.add_argument('--crop-size', type=int, default=128,
                        help='Crop size (must match training)')
    parser.add_argument('--confidence', type=float, default=0.5,
                        help='Confidence threshold')

    args = parser.parse_args()

    test_video(
        video_path=args.video,
        model_path=args.model,
        use_gpu=args.gpu,
        crop_size=args.crop_size,
        confidence_threshold=args.confidence
    )
