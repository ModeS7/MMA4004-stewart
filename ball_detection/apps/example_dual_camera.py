"""
Example: Dual Camera Ball Detection Pipeline

Demonstrates how to use BallDetector for stereo tracking with two cameras.
"""

import cv2
import numpy as np
import time
from ball_detection import BallDetector


def dual_camera_example(model_path, camera1_id=0, camera2_id=1, use_gpu=True):
    """
    Example dual camera detection pipeline.

    Args:
        model_path: Path to trained ONNX model
        camera1_id: Camera 1 device ID
        camera2_id: Camera 2 device ID
        use_gpu: Whether to use DirectML GPU acceleration
    """
    print("=" * 60)
    print("Dual Camera Ball Detection Example")
    print("=" * 60)

    # Initialize detector
    print("\nInitializing detector...")
    detector = BallDetector(
        onnx_model_path=model_path,
        use_gpu=use_gpu,
        crop_size=64,
        confidence_threshold=0.5
    )

    # Open both cameras
    print(f"\nOpening cameras...")
    cap1 = cv2.VideoCapture(camera1_id)
    cap2 = cv2.VideoCapture(camera2_id)

    if not cap1.isOpened():
        print(f"Error: Could not open camera {camera1_id}")
        return

    if not cap2.isOpened():
        print(f"Error: Could not open camera {camera2_id}")
        cap1.release()
        return

    # Optional: Set camera properties for better performance
    for cap in [cap1, cap2]:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 60)

    print("\nCameras ready!")
    print("\nControls:")
    print("  q: Quit")
    print("  s: Print statistics")
    print("\nRunning detection...\n")

    # Performance tracking
    frame_count = 0
    fps_start = time.time()
    detection_times = []

    # Detection tracking
    detections_cam1 = 0
    detections_cam2 = 0
    stereo_pairs = 0

    try:
        while True:
            # Capture frames from both cameras
            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()

            if not (ret1 and ret2):
                print("Warning: Failed to capture from one or both cameras")
                break

            # Detect ball in both cameras (batched for speed)
            det_start = time.time()
            result1, result2 = detector.detect_dual_camera(frame1, frame2)
            detection_time = (time.time() - det_start) * 1000  # ms
            detection_times.append(detection_time)

            # Track statistics
            if result1:
                detections_cam1 += 1
            if result2:
                detections_cam2 += 1
            if result1 and result2:
                stereo_pairs += 1

            # Visualize results
            vis1 = detector.visualize(frame1, result1)
            vis2 = detector.visualize(frame2, result2)

            # Add camera labels
            cv2.putText(vis1, "Camera 1", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(vis2, "Camera 2", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Add detection info
            if result1:
                x1, y1, conf1 = result1
                cv2.putText(vis1, f"Detected: ({x1:.1f}, {y1:.1f})",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                cv2.putText(vis1, "No detection",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            if result2:
                x2, y2, conf2 = result2
                cv2.putText(vis2, f"Detected: ({x2:.1f}, {y2:.1f})",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                cv2.putText(vis2, "No detection",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # Add performance info
            frame_count += 1
            if frame_count % 30 == 0:
                fps = 30 / (time.time() - fps_start)
                fps_start = time.time()
                avg_det_time = np.mean(detection_times[-30:])

                info_text = f"FPS: {fps:.1f} | Det: {avg_det_time:.1f}ms"
                cv2.putText(vis1, info_text, (10, vis1.shape[0] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv2.putText(vis2, info_text, (10, vis2.shape[0] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            # Stack views side by side
            combined = np.hstack([vis1, vis2])

            # Display
            cv2.imshow('Dual Camera Detection', combined)

            # Handle keyboard
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('s'):
                # Print statistics
                print("\n" + "=" * 60)
                print("STATISTICS")
                print("=" * 60)
                print(f"Frames processed: {frame_count}")
                print(f"Camera 1 detections: {detections_cam1} ({detections_cam1/frame_count*100:.1f}%)")
                print(f"Camera 2 detections: {detections_cam2} ({detections_cam2/frame_count*100:.1f}%)")
                print(f"Stereo pairs: {stereo_pairs} ({stereo_pairs/frame_count*100:.1f}%)")

                avg_det = np.mean(detection_times)
                print(f"\nDetection time: {avg_det:.2f} ± {np.std(detection_times):.2f} ms")
                print(f"Min: {np.min(detection_times):.2f} ms")
                print(f"Max: {np.max(detection_times):.2f} ms")
                print(f"P95: {np.percentile(detection_times, 95):.2f} ms")

                stats = detector.get_statistics()
                print(f"\nCNN inferences: {stats['cnn_inferences']}")
                print(f"CNN avg time: {stats['cnn_avg_time_ms']:.2f} ms")
                print(f"Using GPU: {stats['using_gpu']}")
                print("=" * 60 + "\n")

    except KeyboardInterrupt:
        print("\nInterrupted by user")

    finally:
        # Cleanup
        cap1.release()
        cap2.release()
        cv2.destroyAllWindows()

        # Final statistics
        print("\n" + "=" * 60)
        print("FINAL STATISTICS")
        print("=" * 60)
        print(f"Total frames: {frame_count}")
        print(f"Camera 1 detections: {detections_cam1} ({detections_cam1/max(frame_count,1)*100:.1f}%)")
        print(f"Camera 2 detections: {detections_cam2} ({detections_cam2/max(frame_count,1)*100:.1f}%)")
        print(f"Stereo pairs: {stereo_pairs} ({stereo_pairs/max(frame_count,1)*100:.1f}%)")

        if detection_times:
            avg_det = np.mean(detection_times)
            print(f"\nAverage detection time: {avg_det:.2f} ms")
            print(f"Detection throughput: {1000/avg_det:.1f} FPS")

        stats = detector.get_statistics()
        print(f"\nTotal CNN inferences: {stats['cnn_inferences']}")
        print(f"CNN average time: {stats['cnn_avg_time_ms']:.2f} ms")
        print(f"GPU acceleration: {stats['using_gpu']}")
        print("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Dual Camera Detection Example')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to ONNX model')
    parser.add_argument('--cam1', type=int, default=0,
                        help='Camera 1 device ID')
    parser.add_argument('--cam2', type=int, default=1,
                        help='Camera 2 device ID')
    parser.add_argument('--no-gpu', action='store_true',
                        help='Disable GPU acceleration')

    args = parser.parse_args()

    dual_camera_example(
        model_path=args.model,
        camera1_id=args.cam1,
        camera2_id=args.cam2,
        use_gpu=not args.no_gpu
    )
