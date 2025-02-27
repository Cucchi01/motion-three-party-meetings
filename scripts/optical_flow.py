from genericpath import isfile
import os
import constants
from constants import FOLDER_VIDEOS, OpticalFlowConstansts
import cv2
import dlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm
from pathlib import Path
import datetime
import timeit

# face detector able to detect faces that are looking more or less to the camera
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(str(constants.SHAPE_PREDICTOR_PATH))


# Open video file
def compute_motions(
    path: Path, current_num_file: int, tot_num_videos: int, start_time
) -> int:
    if path.is_dir():
        # scan the folder recursively
        for file in path.iterdir():
            current_num_file = compute_motions(
                file, current_num_file, tot_num_videos, start_time
            )
        return current_num_file

    current_num_file += 1

    # output path
    relative_pos_index = path.parts.index(FOLDER_VIDEOS.name)
    feature_path = constants.FOLDER_FEATURES_MOTION.joinpath(
        *path.parts[relative_pos_index + 1 :]
    )
    if not feature_path.parent.exists():
        os.makedirs(feature_path.parent)
    feature_path = feature_path.with_suffix(".csv")

    # features already computed for this video
    if feature_path.exists():
        return current_num_file

    # load video and metadata
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    if current_num_file == 1:
        print(f"File: {path.name}; File number 1 of {tot_num_videos}")
    else:
        estimated_time_left = (
            (tot_num_videos - current_num_file + 1)
            * (timeit.default_timer() - start_time)
            / (current_num_file - 1)
        )
        print(
            f"File: {path.name}; File number {current_num_file} of {tot_num_videos}; Estimated time left:{datetime.timedelta(seconds=(int(estimated_time_left)))}"
        )
    print(f"FPS: {fps}; Num frames: {num_frames}; duration: {num_frames/fps:.2f} s")

    list_magnitudes = []

    # Read the first frame and convert to grayscale
    ret, prev_frame = cap.read()
    print(f"Quality {prev_frame.shape[0]} X {prev_frame.shape[1]}")
    prev_frame = cv2.GaussianBlur(
        prev_frame, (5, 5), 1 / (2 * OpticalFlowConstansts.DOWNSAMPLE_RATE)
    )  # kernel size and standard deviation
    prev_frame = cv2.resize(
        prev_frame,
        (
            prev_frame.shape[1] // OpticalFlowConstansts.DOWNSAMPLE_RATE,
            prev_frame.shape[0] // OpticalFlowConstansts.DOWNSAMPLE_RATE,
        ),
    )
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    # process the remaining frames
    for num_frame in tqdm(
        range(1, int(num_frames), OpticalFlowConstansts.KEEP_EVERY_NUM_FRAMES)
    ):
        if not cap.isOpened():
            break

        # set the position to the next frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, num_frame)
        ret, frame = cap.read()
        if not ret:
            break

        # smooth and downsample
        frame = cv2.GaussianBlur(
            frame, (5, 5), 1 / (2 * OpticalFlowConstansts.DOWNSAMPLE_RATE)
        )  # kernel size and standard deviation
        frame = cv2.resize(
            frame,
            (
                frame.shape[1] // OpticalFlowConstansts.DOWNSAMPLE_RATE,
                frame.shape[0] // OpticalFlowConstansts.DOWNSAMPLE_RATE,
            ),
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

            # Expand box
            padding = OpticalFlowConstansts.PADDING_BOX_BOTTOM_FACE
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
        if (
            OpticalFlowConstansts.VISUALIZE
            and num_frame % OpticalFlowConstansts.VISUALIZE_EVERY_NUM_FRAMES == 0
        ):
            print("Average magnitude: ", list_magnitudes[-1])
            fig, ax = plt.subplots()
            ax.imshow(magnitude)
            plt.show()

        # update previous frame
        prev_gray = gray

    cap.release()

    magnitudes = np.array(list_magnitudes)
    if OpticalFlowConstansts.AVERAGE_BY_WINDOW:
        # computer average motion for each window

        # length in seconds seconds
        window_length = OpticalFlowConstansts.WINDOW_LENGTH
        num_frames_per_window = int(
            window_length * fps / OpticalFlowConstansts.KEEP_EVERY_NUM_FRAMES
        )

        cumsums = np.cumsum(np.insert(magnitudes, 0, 0))
        averages = (
            cumsums[num_frames_per_window:] - cumsums[:-num_frames_per_window]
        ) / float(num_frames_per_window)
        averages = averages[::num_frames_per_window]

        # add last partial window
        if (magnitudes.shape[0]) % num_frames_per_window != 0:
            start_pos = averages.shape[0] * num_frames_per_window
            averages = np.hstack([averages, np.average(magnitudes[start_pos:])])

        timestamps = np.arange(averages.shape[0]) * window_length

        data = pd.DataFrame()
        data["Averages"] = averages
        data["Timestamp"] = timestamps
    else:
        # report each timestamp
        seconds_per_frame = OpticalFlowConstansts.KEEP_EVERY_NUM_FRAMES / fps
        timestamps = np.arange(magnitudes.shape[0]) * seconds_per_frame

        data = pd.DataFrame()
        data["AVG_Optical_Flow"] = magnitudes
        data["Timestamp"] = timestamps

    # save the data
    data.to_csv(feature_path, index=False)

    return current_num_file


total_number_videos = sum(
    len(files) for _, _, files in os.walk(constants.FOLDER_VIDEOS)
)
print(f"Total number of videos: {total_number_videos}")
start = timeit.default_timer()
compute_motions(
    constants.FOLDER_VIDEOS,
    current_num_file=0,
    tot_num_videos=total_number_videos,
    start_time=start,
)
