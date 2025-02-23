import constants
import cv2
import dlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm

# face detector able to detect faces that are looking more or less to the camera
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(str(constants.SHAPE_PREDICTOR_PATH))

# Open video file
path_video = str(constants.FOLDER_VIDEOS) + "/video_test.mp4"

cap = cv2.VideoCapture(path_video)
fps = cap.get(cv2.CAP_PROP_FPS)
num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
print(f"FPS: {fps}; Num frames: {num_frames}; duration: {num_frames/fps:.2f} s")

list_magnitudes = []

visualize = False
visualize_every_num_frames = 30
# do not consider all the frames to lower the amount of computations
keep_every_num_frame = 10
downsample_rate = 4

# Read the first frame and convert to grayscale
ret, prev_frame = cap.read()
prev_frame = cv2.GaussianBlur(
    prev_frame, (5, 5), 1 / (2 * downsample_rate)
)  # kernel size and standard deviation
prev_frame = cv2.resize(
    prev_frame,
    (prev_frame.shape[1] // downsample_rate, prev_frame.shape[0] // downsample_rate),
)
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)


for num_frame in tqdm(range(0, int(num_frames), keep_every_num_frame)):
    if not cap.isOpened():
        break

    # set the position to the next frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, num_frame)
    ret, frame = cap.read()
    if not ret:
        break

    # smooth and downsample
    frame = cv2.GaussianBlur(
        frame, (5, 5), 1 / (2 * downsample_rate)
    )  # kernel size and standard deviation
    frame = cv2.resize(
        frame, (frame.shape[1] // downsample_rate, frame.shape[0] // downsample_rate)
    )

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Compute optical flow (Farneback method)
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
    )  # (prev, next, ., pyramid_scale, num_pyramid_layers, average_window_size, num_pixel_neighborhood_poly_expansion, std_related_to_num_pixel_neigh, flags), pyramid_scale =0.5 is classical pyramid. High average window size get robust results, but more blurred
    magnitude = cv2.magnitude(flow[..., 0], flow[..., 1])

    # Detect faces
    faces = detector(gray)
    for face in faces:
        # Get facial landmarks
        landmarks = predictor(gray, face)

        # Mouth coordinates
        mouth_x1 = landmarks.part(4).x
        mouth_y1 = landmarks.part(51).y
        mouth_x2 = landmarks.part(12).x
        mouth_y2 = landmarks.part(8).y

        # Expand box slightly
        padding = 10
        mouth_x1 -= padding
        mouth_y1 -= padding
        mouth_x2 += padding
        mouth_y2 += padding

        # Draw black box over mouth
        cv2.rectangle(
            magnitude, (mouth_x1, mouth_y1), (mouth_x2, mouth_y2), (0, 0, 0), -1
        )

    list_magnitudes.append(np.average(magnitude))

    # show output
    if visualize and num_frame % visualize_every_num_frames == 0:
        print("Average magnitude: ", list_magnitudes[-1])
        fig, ax = plt.subplots()
        ax.imshow(magnitude)
        plt.show()

    # update previous frame
    prev_gray = gray

cap.release()

# computer average motion for each window

# length in seconds seconds
window_length = 5
num_frames_per_window = int(window_length * fps / keep_every_num_frame)

magnitudes = np.array(list_magnitudes)

print(np.insert(magnitudes, 0, 0))
cumsums = np.cumsum(np.insert(magnitudes, 0, 0))
averages = (cumsums[num_frames_per_window:] - cumsums[:-num_frames_per_window]) / float(
    num_frames_per_window
)
print(averages)
averages = averages[::num_frames_per_window]
timestamps = np.arange(averages.shape[0]) * window_length

data = pd.DataFrame()
data["Averages"] = averages
data["Timestamp"] = timestamps
print(data)
