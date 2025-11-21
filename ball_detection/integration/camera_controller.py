"""
ZED Camera Controller for Ball Detection

Runs ball detection in separate thread using CNN-based detector.
Compatible interface with Pixy2 serial controller.
"""

import cv2
import numpy as np
import time
import threading
from queue import Queue
from typing import Optional, Dict, Any
from pathlib import Path

# Import camera config (ball detector imported later to avoid early onnxruntime load)
import sys
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from core.utils import ZEDCameraConfig


class ZEDCameraController:
    """
    ZED camera controller with CNN ball detection.

    Runs in separate thread, similar to SerialController architecture.
    Uses left camera from stereo pair for ball detection.
    """

    def __init__(self, camera_id: int = ZEDCameraConfig.CAMERA_ID,
                 model_path: str = ZEDCameraConfig.MODEL_PATH,
                 use_gpu: bool = ZEDCameraConfig.USE_GPU):
        """
        Initialize ZED camera controller.

        Args:
            camera_id: OpenCV camera device ID
            model_path: Path to trained ONNX model
            use_gpu: Whether to use GPU acceleration
        """
        self.camera_id = camera_id
        self.model_path = model_path
        self.use_gpu = use_gpu

        # Camera and detector
        self.cap = None
        self.detector = None

        # Camera properties (set during connect)
        self.is_stereo = False
        self.frame_width = 0
        self.frame_height = 0
        self.camera_width = 0
        self.camera_height = 0
        self.center_x_px = 0.0
        self.center_y_px = 0.0
        self.pixels_to_mm_x = 0.0
        self.pixels_to_mm_y = 0.0

        # Threading
        self.running = False
        self.thread = None

        # Ball data queue (same interface as SerialController)
        self.ball_data_queue = Queue(maxsize=10)

        # Statistics
        self.frame_count = 0
        self.detection_count = 0
        self.start_time = None

    def connect(self) -> tuple[bool, str]:
        """
        Connect to ZED camera and initialize detector.

        Returns:
            Tuple of (success: bool, message: str)
        """
        try:
            # Import ball detector here to avoid loading onnxruntime until needed
            from ball_detection.core.detector import BallDetector

            # Open camera with DirectShow backend (better for Windows USB cameras)
            # Use DSHOW to avoid frame buffering issues
            self.cap = cv2.VideoCapture(self.camera_id, cv2.CAP_DSHOW)

            if not self.cap.isOpened():
                # Fallback to default backend
                self.cap = cv2.VideoCapture(self.camera_id)
                if not self.cap.isOpened():
                    return False, f"Could not open camera {self.camera_id}"

            # Reduce buffer size to minimize frame tearing and latency
            # Smaller buffer = fresher frames, less tearing
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            # Set resolution and frame rate BEFORE applying other settings
            # Try ZED stereo mode: 2560x720 @ 60fps
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap.set(cv2.CAP_PROP_FPS, 60)

            # Try MJPEG codec for higher resolution/fps support
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

            # Apply camera settings (disable AUTO features for consistent detection)
            self._apply_camera_settings()

            # Get actual camera properties
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(self.cap.get(cv2.CAP_PROP_FPS))

            print(f"Camera resolution: {width}x{height} @ {fps}fps")

            # Detect if stereo camera (width roughly 2x height)
            self.is_stereo = (width > height * 1.5)
            self.frame_width = width
            self.frame_height = height

            # Calculate dimensions for single camera frame
            if self.is_stereo:
                self.camera_width = width // 2
                self.camera_height = height
                print(f"Detected stereo camera, using {ZEDCameraConfig.STEREO_CAMERA} camera")
            else:
                self.camera_width = width
                self.camera_height = height
                print(f"Detected mono camera")

            # Use calibrated platform center from config (not geometric center)
            self.center_x_px = ZEDCameraConfig.CENTER_X
            self.center_y_px = ZEDCameraConfig.CENTER_Y

            # Calculate pixel-to-mm conversion using config FOV
            self.pixels_to_mm_x = ZEDCameraConfig.FOV_WIDTH_MM / self.camera_width
            self.pixels_to_mm_y = ZEDCameraConfig.FOV_HEIGHT_MM / self.camera_height

            print(f"Single camera frame: {self.camera_width}x{self.camera_height}")
            print(f"Platform center: ({self.center_x_px}, {self.center_y_px}) pixels")
            print(f"Pixel-to-mm conversion: X={self.pixels_to_mm_x:.3f}, Y={self.pixels_to_mm_y:.3f}")

            # Initialize ball detector
            print(f"Initializing ball detector...")
            self.detector = BallDetector(
                onnx_model_path=self.model_path,
                use_gpu=self.use_gpu,
                crop_size=ZEDCameraConfig.CROP_SIZE,
                confidence_threshold=ZEDCameraConfig.CONFIDENCE_THRESHOLD
            )

            # Start detection thread
            self.running = True
            self.thread = threading.Thread(target=self._detection_loop, daemon=True)
            self.thread.start()
            self.start_time = time.time()

            return True, f"Camera connected ({width}x{height} @ {fps}fps)"

        except Exception as e:
            if self.cap:
                self.cap.release()
            return False, f"Camera connection failed: {str(e)}"

    def _apply_camera_settings(self) -> None:
        """Apply camera settings from config file or defaults."""
        import json

        config_file = Path(__file__).parent / "camera_config.json"

        # Default settings optimized for ball detection
        default_settings = {
            'AUTO_EXPOSURE': 0.25,   # Disable auto-exposure (0.25 = manual mode for some cameras)
            'EXPOSURE': -6,          # Manual exposure value (adjust based on lighting)
            'AUTO_WB': 1,            # Enable auto white balance
            'BRIGHTNESS': 5,
            'CONTRAST': 2,
            'SATURATION': 4,
            'GAIN': 2,
            'SHARPNESS': 0,
            'GAMMA': 102
        }

        try:
            if config_file.exists():
                # Load settings from file
                with open(config_file, 'r') as f:
                    config = json.load(f)

                print("Applying saved camera settings:")
                settings = config.get('settings', {})
            else:
                # Use default settings
                print("Applying default camera settings:")
                settings = default_settings

            # Apply settings to camera
            for name, value in settings.items():
                prop_id = getattr(cv2, f'CAP_PROP_{name}', None)
                if prop_id is not None:
                    self.cap.set(prop_id, value)
                    if 'AUTO' in name:
                        print(f"  {name} = {'OFF' if value < 0.5 else 'ON'}")
                    else:
                        print(f"  {name} = {value:.1f}")

        except Exception as e:
            print(f"Warning: Could not apply camera settings: {e}")
            print("Camera will use default automatic settings")

    def disconnect(self) -> None:
        """Disconnect from camera and stop detection thread."""
        self.running = False

        if self.thread:
            self.thread.join(timeout=1.0)

        if self.cap:
            self.cap.release()

        print("ZED camera disconnected")

    def _detection_loop(self) -> None:
        """Main detection loop (runs in separate thread)."""
        while self.running and self.cap and self.cap.isOpened():
            try:
                # Flush old frames from buffer (grab without decode to skip buffered frames)
                # This helps reduce frame tearing and ensures fresh frames
                self.cap.grab()

                # Read fresh frame
                ret, frame = self.cap.read()

                if not ret:
                    print("Warning: Failed to read frame from ZED camera")
                    time.sleep(0.01)
                    continue

                self.frame_count += 1

                # Extract camera frame (stereo or mono)
                if self.is_stereo:
                    # Split stereo frame and extract selected camera
                    width = frame.shape[1]
                    if ZEDCameraConfig.STEREO_CAMERA == 'LEFT':
                        camera_frame = frame[:, :width//2]
                    else:  # RIGHT
                        camera_frame = frame[:, width//2:]
                else:
                    # Mono camera - use full frame
                    camera_frame = frame

                # Run ball detection
                detection_result = self.detector.detect(camera_frame)

                # Convert to ball data format (compatible with Pixy2 interface)
                if detection_result is not None:
                    x_px, y_px, confidence = detection_result

                    # Convert pixel coordinates to mm (from platform center)
                    # X: positive = right, negative = left
                    ball_x_mm = (x_px - self.center_x_px) * self.pixels_to_mm_x

                    # Y: Cartesian coordinates (origin at platform center)
                    # Positive = toward top of image (toward camera)
                    # Negative = toward bottom of image (away from camera)
                    # Note: Pixy2 formula only works when platform is at geometric center
                    ball_y_mm = (self.center_y_px - y_px) * self.pixels_to_mm_y

                    ball_data = {
                        'timestamp': time.time(),
                        'x': ball_x_mm,
                        'y': ball_y_mm,
                        'detected': True,
                        'error_x': 0.0,  # Not used with ZED, kept for compatibility
                        'error_y': 0.0,
                        'confidence': confidence
                    }

                    self.detection_count += 1
                else:
                    # No detection
                    ball_data = {
                        'timestamp': time.time(),
                        'x': 0.0,
                        'y': 0.0,
                        'detected': False,
                        'error_x': 0.0,
                        'error_y': 0.0,
                        'confidence': 0.0
                    }

                # Put in queue (non-blocking, drop oldest if full)
                if self.ball_data_queue.full():
                    try:
                        self.ball_data_queue.get_nowait()
                    except:
                        pass

                self.ball_data_queue.put(ball_data)

            except Exception as e:
                print(f"Error in ZED detection loop: {e}")
                time.sleep(0.01)

    def get_latest_ball_data(self) -> Optional[Dict[str, Any]]:
        """
        Get latest ball detection data.

        Returns:
            Dictionary with ball data, or None if no data available.
            Format matches SerialController interface:
            {
                'timestamp': float,
                'x': float (mm),
                'y': float (mm),
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
            Dictionary with frame count, detection rate, and FPS
        """
        elapsed = time.time() - self.start_time if self.start_time else 0.0
        fps = self.frame_count / elapsed if elapsed > 0 else 0.0
        detection_rate = self.detection_count / max(self.frame_count, 1)

        return {
            'frames': self.frame_count,
            'detections': self.detection_count,
            'detection_rate': detection_rate,
            'fps': fps,
            'elapsed_time': elapsed
        }


if __name__ == "__main__":
    """Test ZED camera controller standalone."""
    print("Testing ZED Camera Controller")
    print("Press Ctrl+C to stop\n")

    controller = ZEDCameraController()
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
                    print(f"Ball detected: x={data['x']:.1f}mm, y={data['y']:.1f}mm, conf={data['confidence']:.3f}")
                else:
                    print("No detection")

            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n\nStopping...")
        stats = controller.get_statistics()
        print(f"\nStatistics:")
        print(f"  Frames: {stats['frames']}")
        print(f"  Detections: {stats['detections']}")
        print(f"  Detection rate: {stats['detection_rate']*100:.1f}%")
        print(f"  FPS: {stats['fps']:.1f}")

        controller.disconnect()
