"""
Stereo Camera Controller for Ball Detection

Runs ball detection on both cameras with stereo triangulation.
Outputs 3D coordinates (X, Y, Z) in platform frame.
Compatible interface with Pixy2 serial controller.
"""

import cv2
import numpy as np
import time
import os
import threading
from multiprocessing import Process, Value
from multiprocessing.shared_memory import SharedMemory
from queue import Queue, Empty
from typing import Optional, Dict, Any, Tuple
from pathlib import Path

import sys
# Add project root to path for core.utils import
# NOTE: Only add project root, NOT ball_detection_dir (it has its own core/ that shadows core.utils)
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from core.utils import (
    StereoCameraConfig, StereoDetectionConfig,
    DEBUG_DETECTION_TIMING, DEBUG_TIMING_INTERVAL, DETECTION_MODE
)
from ball_detection.utils.camera import create_camera_capture, load_camera_config, apply_camera_settings
from ball_detection.utils.calibration import load_stereo_calibration
from ball_detection.utils.coordinate_transform import load_platform_transform, apply_platform_transform
from ball_detection.core.inference_pool import InferencePool


# Frame dimensions for shared memory
FRAME_WIDTH = 2560
FRAME_HEIGHT = 720
FRAME_SIZE = FRAME_WIDTH * FRAME_HEIGHT * 3  # BGR uint8


class SharedFrameBuffer:
    """Zero-copy frame buffer using shared memory."""

    def __init__(self, name: str, create: bool = False):
        if create:
            self.shm = SharedMemory(name=name, create=True, size=FRAME_SIZE)
        else:
            self.shm = SharedMemory(name=name, create=False)
        self.frame = np.ndarray(
            (FRAME_HEIGHT, FRAME_WIDTH, 3),
            dtype=np.uint8,
            buffer=self.shm.buf
        )

    def write(self, frame: np.ndarray):
        """Write frame to shared memory (grabber process)."""
        np.copyto(self.frame, frame)

    def read(self) -> np.ndarray:
        """Read frame from shared memory (main process)."""
        return self.frame.copy()  # Copy to avoid race condition

    def close(self):
        self.shm.close()

    def unlink(self):
        self.shm.unlink()


def _frame_grabber_process(camera_id: int, shm_name: str, frame_ready, frame_consumed,
                           running_flag, grabber_fps_value, config: dict):
    """Frame grabber running in separate process with independent GIL.

    Runs at full camera speed (~60fps), continuously overwriting shared memory.
    Detection loop reads latest frame when ready (no synchronization wait).

    Args:
        camera_id: OpenCV camera device ID
        shm_name: Name of shared memory segment
        frame_ready: Value flag indicating frame is ready
        frame_consumed: Not used (kept for API compatibility)
        running_flag: Value flag to stop the process
        grabber_fps_value: Value to store current grabber FPS
        config: Camera configuration dict
    """
    import cv2
    import time
    import sys

    def log(msg):
        print(msg, flush=True)

    log("[GrabberProcess] Starting...")

    # Attach to shared memory (don't create, main process creates)
    try:
        buffer = SharedFrameBuffer(name=shm_name, create=False)
        log("[GrabberProcess] Attached to shared memory")
    except Exception as e:
        log(f"[GrabberProcess] Failed to attach to shared memory: {e}")
        return

    # Try backends in order - prioritize MSMF which worked in main process
    cap = None
    backend_names = {cv2.CAP_MSMF: 'MSMF', cv2.CAP_DSHOW: 'DirectShow', cv2.CAP_ANY: 'Any'}
    for backend in [cv2.CAP_MSMF, cv2.CAP_DSHOW, cv2.CAP_ANY]:
        log(f"[GrabberProcess] Trying {backend_names.get(backend, backend)}...")
        cap = cv2.VideoCapture(camera_id + backend)
        if cap.isOpened():
            log(f"[GrabberProcess] Opened with {backend_names.get(backend, backend)}")
            break
        else:
            log(f"[GrabberProcess] Failed to open with {backend_names.get(backend, backend)}")

    if not cap or not cap.isOpened():
        log(f"[GrabberProcess] FAILED to open camera {camera_id} with any backend!")
        buffer.close()
        return

    # Configure camera - set FOURCC before resolution (some cameras require this order)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.get('width', 2560))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.get('height', 720))
    cap.set(cv2.CAP_PROP_FPS, config.get('fps', 60))
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # Apply camera settings if provided
    if 'settings' in config:
        for prop, val in config['settings'].items():
            cap.set(prop, val)

    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    actual_bufsize = cap.get(cv2.CAP_PROP_BUFFERSIZE)
    log(f"[GrabberProcess] Camera: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} @ {actual_fps}fps, bufsize={actual_bufsize}")

    # Warm up - discard first few frames which may be stale
    for _ in range(3):
        cap.read()

    # FPS tracking
    fps_window_start = time.perf_counter()
    fps_window_count = 0

    # Timing for debugging
    read_times = []
    copy_times = []

    while running_flag.value:
        t0 = time.perf_counter()
        ret, frame = cap.read()
        t1 = time.perf_counter()

        if not ret or frame is None:
            time.sleep(0.001)
            continue

        # Write to shared memory (continuous, no waiting for consumer)
        buffer.write(frame)
        t2 = time.perf_counter()

        # Signal frame ready
        frame_ready.value = 1

        # Track timing
        read_times.append((t1 - t0) * 1000)
        copy_times.append((t2 - t1) * 1000)

        # Update FPS every 100 frames
        fps_window_count += 1
        if fps_window_count >= 100:
            elapsed = time.perf_counter() - fps_window_start
            current_fps = fps_window_count / elapsed if elapsed > 0 else 0
            grabber_fps_value.value = current_fps
            avg_read = sum(read_times) / len(read_times) if read_times else 0
            avg_copy = sum(copy_times) / len(copy_times) if copy_times else 0
            log(f"[GrabberProcess] {current_fps:.1f}fps | read: {avg_read:.1f}ms | copy: {avg_copy:.1f}ms")
            fps_window_start = time.perf_counter()
            fps_window_count = 0
            read_times = []
            copy_times = []

        # No waiting - run at full camera speed

    cap.release()
    buffer.close()
    log("[GrabberProcess] Stopped")


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

        # Frame grabber thread (for ROI_CNN mode - threading works fine)
        self.grabber_thread = None
        self.frame_lock = threading.Lock()
        self.latest_frame = None
        self.frame_ready_event = threading.Event()
        self.grabber_fps = 0.0

        # Inference pool (for STEREO_NN mode - moves inference to separate process)
        self.inference_pool = None

        # Legacy: Frame grabber process (not used - kept for reference)
        self.grabber_process = None
        self.shm = None
        self.shared_buffer = None
        self.shm_name = None
        # Multiprocessing sync flags (ctypes Value)
        self.mp_running_flag = None
        self.mp_frame_ready = None
        self.mp_frame_consumed = None
        self.mp_grabber_fps = None

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

            # Use MJPG codec (compressed, reliable)
            # YUYV (raw) caused access violations on some cameras
            codecs = [
                ('MJPG', 'MJPG'),
            ]

            for backend, backend_name in backends:
                print(f"Trying {backend_name} backend...")
                for fourcc, codec_name in codecs:
                    cap = cv2.VideoCapture(self.camera_id + backend)
                    if cap.isOpened():
                        # Set FOURCC first (some cameras need codec before resolution)
                        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
                        # Then set resolution and FPS
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, StereoCameraConfig.STEREO_FRAME_WIDTH)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, StereoCameraConfig.STEREO_FRAME_HEIGHT)
                        cap.set(cv2.CAP_PROP_FPS, StereoCameraConfig.TARGET_FPS)

                        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        actual_fps = cap.get(cv2.CAP_PROP_FPS)
                        actual_fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
                        actual_fourcc_str = ''.join([chr((actual_fourcc >> 8 * i) & 0xFF) for i in range(4)])

                        if actual_width >= StereoCameraConfig.STEREO_FRAME_WIDTH * 0.9:
                            self.cap = cap
                            used_backend = backend_name
                            print(f"Using {backend_name} + {codec_name} (actual={actual_fourcc_str}, fps={actual_fps})")
                            break
                        else:
                            print(f"{backend_name} + {codec_name}: Got {actual_width}px width, need {StereoCameraConfig.STEREO_FRAME_WIDTH}")
                            cap.release()
                    else:
                        print(f"{backend_name}: Failed to open")
                        break  # Don't try other codecs if backend failed
                if self.cap is not None:
                    break

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

            # Initialize detection mode
            self.detection_mode = DETECTION_MODE
            print(f"Detection mode: {self.detection_mode}")

            # Start frame grabber FIRST (before initializing detectors)
            # This keeps camera alive during potentially slow GPU initialization
            self.running = True
            self.grabber_thread = threading.Thread(target=self._frame_grabber_loop, daemon=True)
            self.grabber_thread.start()

            # Wait for first frame
            if not self.frame_ready_event.wait(timeout=2.0):
                self.running = False
                self.cap.release()
                return False, "Timeout waiting for first frame from camera"

            print("Grabber thread started, initializing detector...")

            # Now initialize detector (grabber keeps camera alive during GPU init)
            if self.detection_mode == 'STEREO_NN':
                # New pipeline: inference pool (runs in separate process to avoid GIL)
                print(f"Initializing inference pool for stereo detection...")
                self.inference_pool = InferencePool(
                    stereo_model_path=StereoDetectionConfig.STEREO_MODEL_PATH,
                    crop_model_path=StereoDetectionConfig.CROP_MODEL_PATH if StereoDetectionConfig.USE_REFINEMENT else None,
                    use_gpu=StereoDetectionConfig.USE_GPU,
                    confidence_threshold=StereoDetectionConfig.CONFIDENCE_THRESHOLD,
                    use_refinement=StereoDetectionConfig.USE_REFINEMENT,
                    convert_to_rgb=StereoDetectionConfig.CONVERT_TO_RGB
                )
                if not self.inference_pool.start():
                    self.running = False
                    self.cap.release()
                    return False, "Failed to start inference pool"
                self.detector = None  # Using inference_pool instead
            else:  # ROI_CNN
                # Old pipeline: ROI extraction + CNN
                print(f"Initializing ROI + CNN detector...")
                from ball_detection.core.detector import BallDetector
                self.detector = BallDetector(
                    onnx_model_path=self.model_path,
                    use_gpu=self.use_gpu,
                    crop_size=StereoCameraConfig.CROP_SIZE,
                    confidence_threshold=StereoCameraConfig.CONFIDENCE_THRESHOLD
                )

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

    def _cleanup_shared_memory(self) -> None:
        """Clean up shared memory resources."""
        if self.shared_buffer:
            try:
                self.shared_buffer.close()
            except Exception:
                pass
            self.shared_buffer = None

        if self.shm:
            try:
                self.shm.close()
                self.shm.unlink()
            except Exception:
                pass
            self.shm = None

    def disconnect(self) -> None:
        """Disconnect from camera and stop detection thread."""
        self.running = False

        # Stop recording if active
        if self.recording:
            self.stop_recording()

        # Stop detection thread
        if self.thread:
            self.thread.join(timeout=1.0)

        # Stop inference pool (STEREO_NN mode)
        if self.inference_pool:
            self.inference_pool.stop()
            self.inference_pool = None

        # Stop grabber thread
        if self.grabber_thread:
            self.grabber_thread.join(timeout=1.0)
            self.grabber_thread = None

        # Release camera
        if self.cap:
            self.cap.release()
            self.cap = None

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
        fps_window_start = time.perf_counter()
        fps_window_count = 0

        while self.running and self.cap and self.cap.isOpened():
            try:
                ret, frame = self.cap.read()

                if ret and frame is not None:
                    with self.frame_lock:
                        self.latest_frame = frame
                    self.frame_ready_event.set()
                    grab_count += 1
                    fps_window_count += 1

                    # Write frame to video if recording
                    if self.recording and self.video_writer:
                        self.video_writer.write(frame)

                    # Calculate grabber FPS every 100 frames (sliding window)
                    if fps_window_count >= 100:
                        elapsed = time.perf_counter() - fps_window_start
                        self.grabber_fps = fps_window_count / elapsed if elapsed > 0 else 0
                        fps_window_start = time.perf_counter()
                        fps_window_count = 0

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

                # Get frame from threaded grabber (both modes now use this)
                with self.frame_lock:
                    frame = self.latest_frame

                if frame is None:
                    time.sleep(0.001)
                    continue

                t_get_frame = time.perf_counter()

                self.frame_count += 1

                # Split stereo frame into left and right
                width = frame.shape[1]
                left_frame = frame[:, :width//2]
                right_frame = frame[:, width//2:]
                t_split = time.perf_counter()

                # Build ball data
                ball_data = {
                    'timestamp': time.time(),
                    'x': 0.0,
                    'y': 0.0,
                    'z': 0.0,
                    'detected': False,
                    'in_control_zone': False,  # True = within 200mm radius, safe for control
                    'error_x': 0.0,
                    'error_y': 0.0,
                    'confidence': 0.0
                }

                # Run detection based on mode
                if self.detection_mode == 'STEREO_NN':
                    # New pipeline: inference pool (separate process)
                    # Submit frame to pool
                    if not self.inference_pool.submit_frame(frame):
                        continue  # Pool busy, skip this frame

                    # Get result (blocking with timeout)
                    detection_result = self.inference_pool.get_result(timeout=0.05)
                    t_detect = time.perf_counter()

                    if detection_result is None:
                        # No result yet
                        continue

                    timing = detection_result['timing']
                    t_rectify = t_detect

                    if detection_result['detected']:
                        self.stereo_pair_count += 1
                        x_left, y_left = detection_result['x_left'], detection_result['y_left']
                        x_right, y_right = detection_result['x_right'], detection_result['y_right']
                        confidence = detection_result['confidence']

                        # Rectify and triangulate
                        left_rectified = self._rectify_point((x_left, y_left), 'left')
                        right_rectified = self._rectify_point((x_right, y_right), 'right')
                        t_rectify = time.perf_counter()

                        point_3d = self._triangulate_3d_point(left_rectified, right_rectified)
                        if point_3d is not None:
                            self.triangulation_count += 1
                            if self.platform_transform is not None:
                                R, T = self.platform_transform
                                point_3d = apply_platform_transform(point_3d, R, T)

                            # Check if detection is within control zone
                            MAX_RADIUS = 200.0  # mm - circular boundary
                            dist_sq = point_3d[0]**2 + point_3d[1]**2
                            in_control_zone = dist_sq <= MAX_RADIUS**2

                            ball_data = {
                                'timestamp': time.time(),
                                'x': float(point_3d[0]),
                                'y': float(point_3d[1]),
                                'z': float(point_3d[2]),
                                'detected': True,
                                'in_control_zone': in_control_zone,  # False = outside 200mm, don't use for control
                                'error_x': 0.0,
                                'error_y': 0.0,
                                'confidence': confidence
                            }
                            self.detection_count += 1

                    # Debug output for STEREO_NN mode
                    t_end = time.perf_counter()
                    if debug_timing and self.frame_count % timing_interval == 0:
                        total_ms = (t_end - loop_start) * 1000
                        get_frame_ms = (t_get_frame - loop_start) * 1000
                        split_ms = (t_split - t_get_frame) * 1000
                        detect_ms = (t_detect - t_split) * 1000
                        rectify_ms = (t_rectify - t_detect) * 1000
                        rest_ms = (t_end - t_rectify) * 1000
                        print(f"[Frame {self.frame_count}] Total: {total_ms:.1f}ms | "
                              f"get: {get_frame_ms:.1f} | split: {split_ms:.1f} | "
                              f"detect: {detect_ms:.1f} | pt_rect: {rectify_ms:.1f} | rest: {rest_ms:.1f} | "
                              f"grabber: {self.grabber_fps:.1f}fps")
                        print(f"  copy: {timing['copy_ms']:.2f} | prep: {timing['preprocess_ms']:.2f} | "
                              f"stereo_nn: {timing['stereo_ms']:.2f} | "
                              f"refine_L: {timing['refine_L_ms']:.2f} | refine_R: {timing['refine_R_ms']:.2f}")
                        if detection_result['detected']:
                            print(f"  conf={detection_result['confidence']:.2f} | "
                                  f"L: ({detection_result['x_left']:.0f}, {detection_result['y_left']:.0f}) | "
                                  f"R: ({detection_result['x_right']:.0f}, {detection_result['y_right']:.0f})")
                        else:
                            print(f"  No detection (conf={detection_result['confidence']:.2f})")

                else:  # ROI_CNN mode
                    # Old pipeline: ROI extraction + CNN
                    result_left, timing_left = self.detector.detect_with_timing(left_frame)
                    t_detect_left = time.perf_counter()
                    result_right, timing_right = self.detector.detect_with_timing(right_frame)
                    t_detect_right = time.perf_counter()
                    t_rectify = t_detect_right

                    if result_left is not None and result_right is not None:
                        self.stereo_pair_count += 1
                        x_left, y_left, conf_left = result_left
                        x_right, y_right, conf_right = result_right

                        left_rectified = self._rectify_point((x_left, y_left), 'left')
                        right_rectified = self._rectify_point((x_right, y_right), 'right')
                        t_rectify = time.perf_counter()

                        point_3d = self._triangulate_3d_point(left_rectified, right_rectified)
                        if point_3d is not None:
                            self.triangulation_count += 1
                            if self.platform_transform is not None:
                                R, T = self.platform_transform
                                point_3d = apply_platform_transform(point_3d, R, T)

                            # Check if detection is within control zone
                            MAX_RADIUS = 200.0  # mm - circular boundary
                            dist_sq = point_3d[0]**2 + point_3d[1]**2
                            in_control_zone = dist_sq <= MAX_RADIUS**2

                            ball_data = {
                                'timestamp': time.time(),
                                'x': float(point_3d[0]),
                                'y': float(point_3d[1]),
                                'z': float(point_3d[2]),
                                'detected': True,
                                'in_control_zone': in_control_zone,  # False = outside 200mm, don't use for control
                                'error_x': 0.0,
                                'error_y': 0.0,
                                'confidence': min(conf_left, conf_right)
                            }
                            self.detection_count += 1

                    # Debug output for ROI_CNN mode
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
