"""
Stereo Camera Controller for Ball Detection

Runs ball detection on both cameras with stereo triangulation.
Outputs 3D coordinates (X, Y, Z) in platform frame.
Compatible interface with Pixy2 serial controller.
"""

import cv2
import numpy as np
import time
import threading
from queue import Queue, Empty
from typing import Optional, Dict, Any, Tuple
from pathlib import Path

import sys
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from core.utils import StereoCameraConfig, DEBUG_DETECTION_TIMING, DEBUG_TIMING_INTERVAL
from ball_detection.utils.camera import create_camera_capture, load_camera_config, apply_camera_settings
from ball_detection.utils.calibration import load_stereo_calibration
from ball_detection.utils.coordinate_transform import load_platform_transform, apply_platform_transform


class StereoCameraController:
    """
    Stereo camera controller with CNN ball detection and 3D triangulation.

    Runs in separate thread, similar to SerialController architecture.
    Uses both cameras for stereo triangulation to get true 3D coordinates.
    """

    def __init__(self, camera_id: int = StereoCameraConfig.CAMERA_ID,
                 model_path: str = StereoCameraConfig.MODEL_PATH,
                 use_gpu: bool = StereoCameraConfig.USE_GPU,
                 calibration_dir: str = StereoCameraConfig.CALIBRATION_DIR):
        """
        Initialize stereo camera controller.

        Args:
            camera_id: OpenCV camera device ID
            model_path: Path to trained ONNX model
            use_gpu: Whether to use GPU acceleration
            calibration_dir: Directory containing stereo calibration files
        """
        self.camera_id = camera_id
        self.model_path = model_path
        self.use_gpu = use_gpu
        self.calibration_dir = Path(calibration_dir)

        # Camera and detector
        self.cap = None
        self.detector = None

        # Stereo calibration data
        self.stereo_calib = None
        self.platform_transform = None
        self.P1 = None
        self.P2 = None

        # Camera properties (set during connect)
        self.frame_width = 0
        self.frame_height = 0

        # Threading
        self.running = False
        self.thread = None

        # Frame grabber thread (decouples camera latency from detection)
        self.grabber_thread = None
        self.frame_lock = threading.Lock()
        self.latest_frame = None
        self.frame_ready = threading.Event()
        self.grabber_fps = 0.0

        # Ball data queue (same interface as SerialController)
        self.ball_data_queue = Queue(maxsize=10)

        # Statistics
        self.frame_count = 0
        self.detection_count = 0
        self.stereo_pair_count = 0

        # Video recording
        self.video_writer = None
        self.recording = False
        self.record_path = None
        self.triangulation_count = 0
        self.start_time = None

    def connect(self) -> tuple[bool, str]:
        """
        Connect to stereo camera and initialize detector.

        Returns:
            Tuple of (success: bool, message: str)
        """
        try:
            # Import ball detector here to avoid loading onnxruntime until needed
            from ball_detection.core.detector import BallDetector

            # Load stereo calibration (skip rectification maps, use point-only rectification)
            print(f"Loading stereo calibration from {self.calibration_dir}...")
            self.stereo_calib = load_stereo_calibration(self.calibration_dir, load_maps=False)
            if self.stereo_calib is None:
                return False, "Failed to load stereo calibration"

            self.P1 = self.stereo_calib['P1']
            self.P2 = self.stereo_calib['P2']
            self.K1 = self.stereo_calib['K1']
            self.K2 = self.stereo_calib['K2']
            self.D1 = self.stereo_calib['D1']
            self.D2 = self.stereo_calib['D2']
            self.R1 = self.stereo_calib['R1']
            self.R2 = self.stereo_calib['R2']
            print(f"Stereo calibration loaded (timestamp: {self.stereo_calib['timestamp']})")
            print("Using point-only rectification (no full-frame remap)")

            # Load platform transformation
            print("Loading platform transformation...")
            try:
                platform_dir = self.calibration_dir.parent / 'calibrations'
                if not platform_dir.exists():
                    platform_dir = self.calibration_dir
                transform_data = load_platform_transform(str(platform_dir))
                self.platform_transform = (transform_data['R'], transform_data['T'])
                print(f"Platform transform loaded (timestamp: {transform_data['timestamp']})")
            except FileNotFoundError as e:
                print(f"Warning: {e}")
                print("Coordinates will be in camera frame, not platform frame")
                self.platform_transform = None

            # Try different backends for best latency
            # CAP_DSHOW (DirectShow) often has lower latency on Windows
            backends = [
                (cv2.CAP_DSHOW, "DirectShow"),
                (cv2.CAP_MSMF, "Media Foundation"),
                (cv2.CAP_ANY, "Auto")
            ]

            self.cap = None
            used_backend = "Unknown"

            for backend, name in backends:
                print(f"Trying {name} backend...")
                cap = cv2.VideoCapture(self.camera_id + backend)
                if cap.isOpened():
                    # Test if we can get stereo resolution
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, StereoCameraConfig.STEREO_FRAME_WIDTH)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, StereoCameraConfig.STEREO_FRAME_HEIGHT)
                    cap.set(cv2.CAP_PROP_FPS, StereoCameraConfig.TARGET_FPS)
                    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

                    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    if actual_width >= StereoCameraConfig.STEREO_FRAME_WIDTH * 0.9:
                        self.cap = cap
                        used_backend = name
                        print(f"Using {name} backend")
                        break
                    else:
                        print(f"{name}: Got {actual_width}px width, need {StereoCameraConfig.STEREO_FRAME_WIDTH}")
                        cap.release()
                else:
                    print(f"{name}: Failed to open")

            if self.cap is None:
                return False, f"Could not open camera {self.camera_id} with any backend"

            # Reduce buffer size to minimize frame tearing and latency
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            # Apply camera settings (disable AUTO features for consistent detection)
            self._apply_camera_settings()

            # Get actual camera properties
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(self.cap.get(cv2.CAP_PROP_FPS))

            print(f"Camera resolution: {width}x{height} @ {fps}fps")

            # Verify stereo mode
            if width < StereoCameraConfig.STEREO_FRAME_WIDTH * 0.9:
                return False, f"Camera not in stereo mode. Expected {StereoCameraConfig.STEREO_FRAME_WIDTH}x{StereoCameraConfig.STEREO_FRAME_HEIGHT}, got {width}x{height}"

            self.frame_width = width
            self.frame_height = height

            # Initialize ball detector
            print(f"Initializing ball detector...")
            self.detector = BallDetector(
                onnx_model_path=self.model_path,
                use_gpu=self.use_gpu,
                crop_size=StereoCameraConfig.CROP_SIZE,
                confidence_threshold=StereoCameraConfig.CONFIDENCE_THRESHOLD
            )

            # Start frame grabber thread (continuously reads frames)
            self.running = True
            self.grabber_thread = threading.Thread(target=self._frame_grabber_loop, daemon=True)
            self.grabber_thread.start()

            # Wait for first frame
            if not self.frame_ready.wait(timeout=2.0):
                self.running = False
                self.cap.release()
                return False, "Timeout waiting for first frame from camera"

            # Start detection thread
            self.thread = threading.Thread(target=self._detection_loop, daemon=True)
            self.thread.start()
            self.start_time = time.time()

            frame_type = "platform" if self.platform_transform else "camera"
            return True, f"Stereo camera connected ({width}x{height} @ {fps}fps, {frame_type} frame)"

        except Exception as e:
            if self.cap:
                self.cap.release()
            return False, f"Stereo camera connection failed: {str(e)}"

    def _apply_camera_settings(self) -> None:
        """Apply camera settings from config file or defaults."""
        camera_config = load_camera_config()
        apply_camera_settings(self.cap, camera_config)

    def disconnect(self) -> None:
        """Disconnect from camera and stop detection thread."""
        self.running = False

        # Stop recording if active
        if self.recording:
            self.stop_recording()

        if self.thread:
            self.thread.join(timeout=1.0)

        if self.grabber_thread:
            self.grabber_thread.join(timeout=1.0)

        if self.cap:
            self.cap.release()

        print("Stereo camera disconnected")

    def start_recording(self, output_path: str = None) -> str:
        """Start recording video to file."""
        if self.recording:
            return self.record_path

        if output_path is None:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"recording_{timestamp}.mp4"

        self.record_path = output_path
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 60.0
        frame_size = (self.frame_width, self.frame_height)
        self.video_writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
        self.recording = True
        print(f"Recording started: {output_path}")
        return output_path

    def stop_recording(self) -> None:
        """Stop recording video."""
        if not self.recording:
            return

        self.recording = False
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
        print(f"Recording stopped: {self.record_path}")

    def _triangulate_3d_point(self, left_point: tuple, right_point: tuple) -> Optional[np.ndarray]:
        """
        Triangulate 3D point from corresponding 2D points.

        Args:
            left_point: (x, y) coordinates in left camera
            right_point: (x, y) coordinates in right camera

        Returns:
            3D point [x, y, z] in mm, or None if triangulation fails
        """
        if left_point is None or right_point is None:
            return None

        # Convert to format expected by cv2.triangulatePoints
        left_pt = np.array([[left_point[0]], [left_point[1]]], dtype=np.float32)
        right_pt = np.array([[right_point[0]], [right_point[1]]], dtype=np.float32)

        # Triangulate using projection matrices
        points_4d = cv2.triangulatePoints(self.P1, self.P2, left_pt, right_pt)

        # Convert from homogeneous to 3D coordinates
        points_3d = points_4d[:3] / points_4d[3]

        return points_3d.flatten()

    def _rectify_point(self, point: tuple, camera: str) -> tuple:
        """
        Rectify a single 2D point coordinate.

        Applies lens undistortion and rectification rotation to transform
        a point from raw image coordinates to rectified coordinates.

        Args:
            point: (x, y) coordinates in raw image
            camera: 'left' or 'right'

        Returns:
            (x, y) coordinates in rectified image space
        """
        if camera == 'left':
            K, D, R, P = self.K1, self.D1, self.R1, self.P1
        else:
            K, D, R, P = self.K2, self.D2, self.R2, self.P2

        # Format point for cv2.undistortPoints: shape (1, 1, 2)
        pts = np.array([[[point[0], point[1]]]], dtype=np.float32)

        # Undistort and apply rectification
        rectified = cv2.undistortPoints(pts, K, D, R=R, P=P)

        return (float(rectified[0, 0, 0]), float(rectified[0, 0, 1]))

    def _frame_grabber_loop(self) -> None:
        """Continuously grab frames from camera (runs in separate thread).

        This decouples camera USB latency from detection timing.
        The detection loop always gets the latest frame instantly.
        """
        grab_count = 0
        grab_start = time.perf_counter()

        while self.running and self.cap and self.cap.isOpened():
            try:
                ret, frame = self.cap.read()

                if ret and frame is not None:
                    with self.frame_lock:
                        self.latest_frame = frame
                    self.frame_ready.set()
                    grab_count += 1

                    # Write frame to video if recording
                    if self.recording and self.video_writer:
                        self.video_writer.write(frame)

                    # Calculate grabber FPS every 100 frames
                    if grab_count % 100 == 0:
                        elapsed = time.perf_counter() - grab_start
                        self.grabber_fps = grab_count / elapsed if elapsed > 0 else 0

            except Exception as e:
                print(f"Error in frame grabber: {e}")
                time.sleep(0.001)

    def _detection_loop(self) -> None:
        """Main detection loop (runs in separate thread)."""
        # Timing debug
        debug_timing = DEBUG_DETECTION_TIMING
        timing_interval = DEBUG_TIMING_INTERVAL

        while self.running:
            try:
                loop_start = time.perf_counter()

                # Get latest frame from grabber (non-blocking)
                with self.frame_lock:
                    frame = self.latest_frame
                t_get_frame = time.perf_counter()

                if frame is None:
                    time.sleep(0.001)
                    continue

                self.frame_count += 1

                # Split stereo frame into left and right
                width = frame.shape[1]
                left_frame = frame[:, :width//2]
                right_frame = frame[:, width//2:]
                t_split = time.perf_counter()

                # Run ball detection on raw (unrectified) frames
                result_left, timing_left = self.detector.detect_with_timing(left_frame)
                t_detect_left = time.perf_counter()
                result_right, timing_right = self.detector.detect_with_timing(right_frame)
                t_detect_right = time.perf_counter()
                t_rectify = t_detect_right  # Default (no rectification if no detection)

                # Build ball data
                ball_data = {
                    'timestamp': time.time(),
                    'x': 0.0,
                    'y': 0.0,
                    'z': 0.0,
                    'detected': False,
                    'error_x': 0.0,
                    'error_y': 0.0,
                    'confidence': 0.0
                }

                # Check if we have stereo detection
                if result_left is not None and result_right is not None:
                    self.stereo_pair_count += 1

                    x_left, y_left, conf_left = result_left
                    x_right, y_right, conf_right = result_right

                    # Rectify detected point coordinates (instead of full-frame rectification)
                    left_rectified = self._rectify_point((x_left, y_left), 'left')
                    right_rectified = self._rectify_point((x_right, y_right), 'right')
                    t_rectify = time.perf_counter()

                    # Triangulate 3D point using rectified coordinates
                    point_3d = self._triangulate_3d_point(
                        left_rectified,
                        right_rectified
                    )

                    if point_3d is not None:
                        self.triangulation_count += 1

                        # Apply platform transformation if available
                        if self.platform_transform is not None:
                            R, T = self.platform_transform
                            point_3d = apply_platform_transform(point_3d, R, T)

                        ball_data = {
                            'timestamp': time.time(),
                            'x': float(point_3d[0]),
                            'y': float(point_3d[1]),
                            'z': float(point_3d[2]),
                            'detected': True,
                            'error_x': 0.0,
                            'error_y': 0.0,
                            'confidence': min(conf_left, conf_right)
                        }
                        self.detection_count += 1

                # Timing debug output
                t_end = time.perf_counter()
                if debug_timing and self.frame_count % timing_interval == 0:
                    total_ms = (t_end - loop_start) * 1000
                    get_frame_ms = (t_get_frame - loop_start) * 1000
                    split_ms = (t_split - t_get_frame) * 1000
                    detect_l_ms = (t_detect_left - t_split) * 1000
                    detect_r_ms = (t_detect_right - t_detect_left) * 1000
                    rectify_ms = (t_rectify - t_detect_right) * 1000
                    rest_ms = (t_end - t_rectify) * 1000

                    print(f"[Frame {self.frame_count}] Total: {total_ms:.1f}ms | "
                          f"get: {get_frame_ms:.1f} | split: {split_ms:.1f} | "
                          f"detect_L: {detect_l_ms:.1f} | detect_R: {detect_r_ms:.1f} | "
                          f"pt_rect: {rectify_ms:.1f} | rest: {rest_ms:.1f} | grabber: {self.grabber_fps:.1f}fps")
                    print(f"  L: roi={timing_left['roi_ms']:.1f} prep={timing_left['preprocess_ms']:.1f} cnn={timing_left['inference_ms']:.1f} | "
                          f"R: roi={timing_right['roi_ms']:.1f} prep={timing_right['preprocess_ms']:.1f} cnn={timing_right['inference_ms']:.1f}")

                # Put in queue (non-blocking, drop oldest if full)
                if self.ball_data_queue.full():
                    try:
                        self.ball_data_queue.get_nowait()
                    except:
                        pass

                self.ball_data_queue.put(ball_data)

            except Exception as e:
                print(f"Error in stereo detection loop: {e}")
                time.sleep(0.01)

    def get_latest_ball_data(self) -> Optional[Dict[str, Any]]:
        """
        Get latest ball detection data.

        Returns:
            Dictionary with ball data, or None if no data available.
            Format:
            {
                'timestamp': float,
                'x': float (mm, platform frame),
                'y': float (mm, platform frame),
                'z': float (mm, platform frame, for logging),
                'detected': bool,
                'error_x': float (compatibility),
                'error_y': float (compatibility),
                'confidence': float (CNN confidence)
            }
        """
        try:
            # Get most recent data (non-blocking)
            data = None
            while not self.ball_data_queue.empty():
                data = self.ball_data_queue.get_nowait()
            return data
        except:
            return None

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get detection statistics.

        Returns:
            Dictionary with frame count, detection rates, and FPS
        """
        elapsed = time.time() - self.start_time if self.start_time else 0.0
        fps = self.frame_count / elapsed if elapsed > 0 else 0.0
        detection_rate = self.detection_count / max(self.frame_count, 1)
        stereo_rate = self.stereo_pair_count / max(self.frame_count, 1)

        return {
            'frames': self.frame_count,
            'detections': self.detection_count,
            'stereo_pairs': self.stereo_pair_count,
            'triangulations': self.triangulation_count,
            'detection_rate': detection_rate,
            'stereo_rate': stereo_rate,
            'fps': fps,
            'grabber_fps': self.grabber_fps,
            'elapsed_time': elapsed
        }


if __name__ == "__main__":
    """Test stereo camera controller standalone."""
    print("Testing Stereo Camera Controller")
    print("Press Ctrl+C to stop\n")

    controller = StereoCameraController()
    success, msg = controller.connect()

    if not success:
        print(f"Error: {msg}")
        exit(1)

    print(f"Success: {msg}\n")

    try:
        while True:
            data = controller.get_latest_ball_data()

            if data:
                if data['detected']:
                    print(f"Ball: X={data['x']:.1f}  Y={data['y']:.1f}  Z={data['z']:.1f} mm  conf={data['confidence']:.3f}")
                else:
                    print("No detection")

            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n\nStopping...")
        stats = controller.get_statistics()
        print(f"\nStatistics:")
        print(f"  Frames: {stats['frames']}")
        print(f"  Stereo pairs: {stats['stereo_pairs']}")
        print(f"  Triangulations: {stats['triangulations']}")
        print(f"  Detection rate: {stats['detection_rate']*100:.1f}%")
        print(f"  Stereo rate: {stats['stereo_rate']*100:.1f}%")
        print(f"  Detection FPS: {stats['fps']:.1f}")
        print(f"  Grabber FPS: {stats['grabber_fps']:.1f}")

        controller.disconnect()
