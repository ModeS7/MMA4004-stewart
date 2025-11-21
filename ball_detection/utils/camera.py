"""
Camera Utilities

Common camera initialization and configuration functions used across the ball detection system.
Extracted to reduce code duplication across 8+ files.
"""

import cv2
from typing import Optional, Dict, Any


# Default camera settings for consistent capture
# These settings disable auto-exposure and auto white balance for stable tracking
DEFAULT_CAMERA_SETTINGS = {
    'AUTO_EXPOSURE': 0.25,  # 0.25 = manual mode, 0.75 = auto mode
    'EXPOSURE': -6,          # Manual exposure value (log scale, -13 to -1)
    'AUTO_WB': 0,            # Disable auto white balance
    'WB_TEMPERATURE': 4600,  # Manual white balance (Kelvin)
    'BRIGHTNESS': 128,       # Brightness (0-255)
    'CONTRAST': 128,         # Contrast (0-255)
    'SATURATION': 128,       # Saturation (0-255)
    'GAIN': 0,               # Gain/ISO (0-100)
}


def create_camera_capture(camera_id: int = 0, backend_priority: list = None) -> Optional[cv2.VideoCapture]:
    """
    Create VideoCapture object with fallback backends.

    Args:
        camera_id: Camera device index
        backend_priority: List of cv2.CAP_* backend constants to try in order.
                          Defaults to [CAP_MSMF, CAP_DSHOW] on Windows.

    Returns:
        cv2.VideoCapture object if successful, None otherwise
    """
    if backend_priority is None:
        # Default Windows backends (MSMF is faster, DSHOW is more compatible)
        backend_priority = [cv2.CAP_MSMF, cv2.CAP_DSHOW]

    # Try each backend in priority order
    for backend in backend_priority:
        try:
            cap = cv2.VideoCapture(camera_id, backend)
            if cap.isOpened():
                return cap
            cap.release()
        except Exception:
            continue

    # Final fallback: default backend
    cap = cv2.VideoCapture(camera_id)
    return cap if cap.isOpened() else None


def apply_camera_settings(cap: cv2.VideoCapture, settings: Dict[str, Any]) -> None:
    """
    Apply camera settings from a dictionary.

    Args:
        cap: OpenCV VideoCapture object
        settings: Dictionary mapping setting names to values
                  Example: {'AUTO_EXPOSURE': 0.25, 'EXPOSURE': -6, ...}
    """
    for name, value in settings.items():
        # Convert setting name to cv2.CAP_PROP_* constant
        prop_id = getattr(cv2, f'CAP_PROP_{name}', None)
        if prop_id is not None:
            cap.set(prop_id, value)


def create_dual_camera_capture(camera_id: int = 0,
                                width: int = 2560,
                                height: int = 720,
                                fps: int = 60,
                                apply_defaults: bool = True) -> Optional[cv2.VideoCapture]:
    """
    Create and configure camera capture for dual camera setup (side-by-side stereo).

    Args:
        camera_id: Camera device index
        width: Frame width (2560 for dual 1280x720)
        height: Frame height
        fps: Target frame rate
        apply_defaults: Whether to apply DEFAULT_CAMERA_SETTINGS

    Returns:
        Configured cv2.VideoCapture object or None
    """
    cap = create_camera_capture(camera_id)

    if cap is None:
        return None

    # Configure resolution and performance settings
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize latency
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))  # MJPEG for high resolution

    # Apply default camera settings for consistent capture
    if apply_defaults:
        apply_camera_settings(cap, DEFAULT_CAMERA_SETTINGS)

    return cap


def get_camera_info(cap: cv2.VideoCapture) -> Dict[str, Any]:
    """
    Get current camera configuration.

    Args:
        cap: OpenCV VideoCapture object

    Returns:
        Dictionary with current camera settings
    """
    return {
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'fps': int(cap.get(cv2.CAP_PROP_FPS)),
        'exposure': cap.get(cv2.CAP_PROP_EXPOSURE),
        'auto_exposure': cap.get(cv2.CAP_PROP_AUTO_EXPOSURE),
        'wb_temperature': int(cap.get(cv2.CAP_PROP_WB_TEMPERATURE)),
        'backend': cap.getBackendName()
    }


if __name__ == "__main__":
    # Test camera utilities
    print("Testing camera utilities...")

    cap = create_dual_camera_capture(camera_id=0)

    if cap is not None:
        info = get_camera_info(cap)
        print("\nCamera Info:")
        for key, value in info.items():
            print(f"  {key}: {value}")

        ret, frame = cap.read()
        if ret:
            print(f"\nFrame captured: {frame.shape}")
            print("Camera utilities working correctly!")
        else:
            print("\nWarning: Could not capture frame")

        cap.release()
    else:
        print("Error: Could not open camera")
