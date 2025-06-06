# This script tests opening and reading frames from cameras.
import cv2

def test_cameras(indices):
    caps = []
    frames = []
    for i in indices:
        cap = cv2.VideoCapture(i)
        if not cap.isOpened():
            print(f"Failed to open camera {i}")
            continue

        actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        fourcc_str = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])

        print(f"Camera {i} properties:")
        print(f"  Resolution: {int(actual_width)}x{int(actual_height)}")
        print(f"  FPS: {actual_fps}")
        print(f"  FOURCC: {fourcc_str} {fourcc=}")

        ret, frame = cap.read()
        if not ret:
            print(f"Camera {i} opened but failed to read frame")
            frames.append((i, None))
        else:
            print(f"Camera {i} opened and frame read OK, shape={frame.shape}")
            frames.append((i, frame))
        caps.append(cap)

    # After reading all frames, display them
    for i, frame in frames:
        if frame is not None:
            window_name = f"Camera {i}"
            cv2.imshow(window_name, frame)
            print(f"Press any key in the image window to close Camera {i}")
            cv2.waitKey(0)
            cv2.destroyWindow(window_name)
        else:
            print(f"No frame to display for Camera {i}")
    
    for cap in caps:
        cap.release()

# Example: let's just try 0,1,2 (edit as needed)
camera_indices = [0, 1, 2]

print("=== Testing all 3 cameras on the same hub ===")
test_cameras(camera_indices)