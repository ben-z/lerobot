

import cv2

camera_idx = 0
backend = cv2.CAP_AVFOUNDATION

cv2.setNumThreads(1)
camera = cv2.VideoCapture(camera_idx, backend)
print("is opened:", camera.isOpened())
print("backend:", camera.getBackendName())
print("threads:", cv2.getNumThreads())
print("fps", camera.get(cv2.CAP_PROP_FPS))
default_width = int(round(camera.get(cv2.CAP_PROP_FRAME_WIDTH)))
default_height = int(round(camera.get(cv2.CAP_PROP_FRAME_HEIGHT)))
print("default width:", default_width)
print("default height:", default_height)

print("set fps:", camera.set(cv2.CAP_PROP_FPS, 30))
actual_fps = camera.get(cv2.CAP_PROP_FPS)
print(f"Actual FPS: {actual_fps}")

print("set width:", camera.set(cv2.CAP_PROP_FRAME_WIDTH, 800))
print("set height:", camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 600))

actual_width = camera.get(cv2.CAP_PROP_FRAME_WIDTH)
print(f"Actual width: {actual_width}")
actual_height = camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(f"Actual height: {actual_height}")
