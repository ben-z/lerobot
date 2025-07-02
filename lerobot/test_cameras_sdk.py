# Tests cameras using the lerobot SDK

import time

import cv2

from lerobot.common.cameras.configs import ColorMode, Cv2Rotation
from lerobot.common.cameras.opencv.camera_opencv import OpenCVCamera
from lerobot.common.cameras.opencv.configuration_opencv import OpenCVCameraConfig

camera_params = [
    # (0, 1920, 1080, 5, Cv2Rotation.NO_ROTATION),
    (0, 800, 600, 30, Cv2Rotation.NO_ROTATION),
    # (1, 1920, 1080, 30, Cv2Rotation.NO_ROTATION),
    # (2, 1080, 1920, 30, Cv2Rotation.ROTATE_270),
]

# Construct an `OpenCVCameraConfig` with your desired FPS, resolution, color mode, and rotation.
configs = [OpenCVCameraConfig(
    index_or_path=idx,
    fps=fps,
    width=width,
    height=height,
    color_mode=ColorMode.RGB,
    rotation=rotation
) for idx, width, height, fps, rotation in camera_params]

# Instantiate and connect an `OpenCVCamera`, performing a warm-up read (default).
cameras = [OpenCVCamera(config) for config in configs]
for camera in cameras:
    camera.connect()

time.sleep(10)

# Read frames asynchronously in a loop via `async_read(timeout_ms)`
try:
    for idx, _, _, _, _ in camera_params:
        frame = cameras[idx].async_read(timeout_ms=200)
        print(f"Camera {idx} async frame shape:", frame.shape)
        cv2.imshow(f"Camera {idx}", frame)
        print(f"Press any key in the image window to close Camera {idx}")
        cv2.waitKey(0)
finally:
    for camera in cameras:
        camera.disconnect()
    cv2.destroyAllWindows()