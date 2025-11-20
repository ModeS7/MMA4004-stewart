#!/usr/bin/env python3
"""
Record Dual Camera Video for Training Data Collection

Records from ZED/dual camera at 2560x720 @ 60fps.
Simple script to ensure correct resolution and frame rate.

Usage:
    python -m ball_detection.tools.record_video
    python -m ball_detection.tools.record_video --output my_video.mp4 --duration 300
"""

import cv2
import numpy as np
import argparse
from pathlib import Path
from datetime import datetime
import time


def create_camera_capture(camera_index):
    """Create VideoCapture object with Windows backends."""
    backends = [cv2.CAP_MSMF, cv2.CAP_DSHOW]

    for backend in backends:
        try:
            cap = cv2.VideoCapture(camera_index, backend)
            if cap.isOpened():
                return cap
            cap.release()
        except Exception:
            continue

    cap = cv2.VideoCapture(camera_index)
    return cap if cap.isOpened() else None


def record_video(camera_id=0, output_path=None, duration=None, fps=60):
    """
    Record dual camera video at 2560x720 @ 60fps.

    Args:
        camera_id: Camera device index
        output_path: Output video file path
        duration: Recording duration in seconds (None = until 'q' pressed)
        fps: Target FPS
    """
    print("=" * 60)
    print("Dual Camera Video Recording")
    print("=" * 60)

    # Open camera
    print(f"\nOpening camera {camera_id}...")
    cap = create_camera_capture(camera_id)
    if cap is None:
        print(f"ERROR: Failed to open camera {camera_id}")
        return False

    # Configure camera for dual mode
    print("Configuring camera for 2560x720 @ 60fps...")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, fps)

    # Verify settings
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"\nCamera settings:")
    print(f"  Resolution: {actual_width}x{actual_height}")
    print(f"  FPS: {actual_fps:.1f}")

    if actual_width != 2560 or actual_height != 720:
        print(f"\nWARNING: Expected 2560x720, got {actual_width}x{actual_height}")
        print("Camera may not be in dual mode!")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            cap.release()
            return False

    # Setup output file
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(f"ball_training_{timestamp}.mp4")
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Video codec (use MP4V for compatibility)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (actual_width, actual_height))

    if not writer.isOpened():
        print(f"ERROR: Failed to open video writer for {output_path}")
        cap.release()
        return False

    print(f"\nRecording to: {output_path}")
    if duration:
        print(f"Duration: {duration} seconds")
    else:
        print("Duration: Until 'q' pressed")

    print("\nControls:")
    print("  'q' - Stop recording")
    print("  's' - Show statistics")
    print("\n" + "=" * 60)
    print("RECORDING...")
    print("=" * 60 + "\n")

    # Recording stats
    frame_count = 0
    start_time = time.time()
    last_fps_time = start_time
    dropped_frames = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print(f"\nWarning: Failed to capture frame {frame_count}")
                dropped_frames += 1
                continue

            frame_count += 1

            # Write frame
            writer.write(frame)

            # Create preview (scaled down for display)
            preview = cv2.resize(frame, (1280, 360))

            # Add recording indicator (red circle)
            cv2.circle(preview, (20, 20), 10, (0, 0, 255), -1)
            cv2.putText(preview, "REC", (40, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 255), 2)

            # Add stats
            elapsed = time.time() - start_time
            cv2.putText(preview, f"Frames: {frame_count}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(preview, f"Time: {elapsed:.1f}s", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            if duration:
                remaining = duration - elapsed
                cv2.putText(preview, f"Remaining: {remaining:.1f}s", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Calculate current FPS
            if frame_count % 30 == 0:
                current_time = time.time()
                current_fps = 30 / (current_time - last_fps_time)
                last_fps_time = current_time

                cv2.putText(preview, f"FPS: {current_fps:.1f}", (10, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            # Show preview
            cv2.imshow('Recording Preview (Press Q to stop)', preview)

            # Handle keyboard
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print("\nStopping recording...")
                break
            elif key == ord('s'):
                print(f"\nStatistics:")
                print(f"  Frames recorded: {frame_count}")
                print(f"  Elapsed time: {elapsed:.1f}s")
                print(f"  Average FPS: {frame_count/elapsed:.1f}")
                if dropped_frames > 0:
                    print(f"  Dropped frames: {dropped_frames}")

            # Check duration
            if duration and elapsed >= duration:
                print(f"\nReached target duration of {duration}s")
                break

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")

    finally:
        # Cleanup
        total_time = time.time() - start_time
        cap.release()
        writer.release()
        cv2.destroyAllWindows()

        # Final report
        print("\n" + "=" * 60)
        print("Recording Complete")
        print("=" * 60)
        print(f"Output file: {output_path}")
        print(f"Total frames: {frame_count}")
        print(f"Duration: {total_time:.1f}s")
        print(f"Average FPS: {frame_count/total_time:.1f}")
        print(f"File size: {output_path.stat().st_size / (1024*1024):.1f} MB")

        if dropped_frames > 0:
            print(f"Dropped frames: {dropped_frames}")

        print("\nNext step: Label the data")
        print(f"  python -m ball_detection.tools.labeling_gui {output_path}")
        print("=" * 60)

        return True


def main():
    parser = argparse.ArgumentParser(
        description='Record dual camera video at 2560x720 @ 60fps'
    )

    parser.add_argument('--camera', type=int, default=0,
                        help='Camera device index (default: 0)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output video file path (default: ball_training_TIMESTAMP.mp4)')
    parser.add_argument('--duration', type=int, default=None,
                        help='Recording duration in seconds (default: until q pressed)')
    parser.add_argument('--fps', type=int, default=60,
                        help='Target FPS (default: 60)')

    args = parser.parse_args()

    success = record_video(
        camera_id=args.camera,
        output_path=args.output,
        duration=args.duration,
        fps=args.fps
    )

    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
