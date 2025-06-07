
import cv2

camera_idx = 0

print("--------")
print("write width, read width, write height, read height")
camera = cv2.VideoCapture(camera_idx)
print("set width:", camera.set(cv2.CAP_PROP_FRAME_WIDTH, 800))
print(f"Actual width: {camera.get(cv2.CAP_PROP_FRAME_WIDTH)}")
print("set height:", camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 600))
print(f"Actual height: {camera.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
camera.release()

print("--------")
print("write width, write height, read width, read height")
camera = cv2.VideoCapture(camera_idx)
print("set width:", camera.set(cv2.CAP_PROP_FRAME_WIDTH, 800))
print("set height:", camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 600))
print(f"Actual width: {camera.get(cv2.CAP_PROP_FRAME_WIDTH)}")
print(f"Actual height: {camera.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
camera.release()