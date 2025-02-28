import os
import constants
from constants import FOLDER_VIDEOS_ORIGINAL
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import datetime
import timeit

paths_video_1 = ["SESS002_digital_DESERT_br.mp4", "SESS002_digital_DESERT_bl.mp4"]
paths_video_2 = ["SESS002_digital_DESERT_bc.mp4", "SESS002_digital_DESERT_tr.mp4"]
# seconds where the transition happens (exact frame adjusted below)
seconds = [53.3, 53.3]

# load video and metadata
for path1, path2, second in zip(paths_video_1, paths_video_2, seconds):
    path1 = constants.FOLDER_VIDEOS.joinpath(path1)
    path2 = constants.FOLDER_VIDEOS.joinpath(path2)

    print(f"Joining {path1.name} and {path2.name}")

    # combine the two path for the output path
    path_out = path1.name.split(".")[0] + "_" + path2.name.split("_")[-1]
    path_out = path1.parent.joinpath(path_out)

    # if already processed skip
    if path_out.exists():
        continue

    cap1 = cv2.VideoCapture(str(path1))
    cap2 = cv2.VideoCapture(str(path2))

    # two videos assumed with same duration and fps since they are different channels of the same video
    fps = cap1.get(cv2.CAP_PROP_FPS)
    num_frames = cap1.get(cv2.CAP_PROP_FRAME_COUNT)
    width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if not cap1.isOpened() or not cap2.isOpened():
        continue

    # create mp4 output
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_video = cv2.VideoWriter(str(path_out), fourcc, fps, (width, height))

    # search the transition in a window around the moment set
    window = 5
    start_frame = int((second - window / 2) * fps)
    end_frame = int((second + window / 2) * fps)

    cap1.set(cv2.CAP_PROP_POS_FRAMES, start_frame - 1)
    red, old_frame = cap1.read()
    if not red:
        break

    # transition detect when the maximum number of frames changes in that window
    max_diff = 0
    frame_transition = start_frame
    for num_frame in range(start_frame, end_frame):
        red, frame = cap1.read()
        if not red:
            break

        diff = np.abs(frame - old_frame).sum()
        if diff > max_diff:
            max_diff = diff
            frame_transition = num_frame

        old_frame = frame

    cap1.set(cv2.CAP_PROP_POS_FRAMES, 0)
    cap = cap1

    for num_frame in tqdm(range(0, int(num_frames))):
        if num_frame == frame_transition:
            cap = cap2
            cap2.set(cv2.CAP_PROP_POS_FRAMES, num_frame)

        ret, frame = cap.read()
        if not ret:
            break

        output_video.write(frame)

    output_video.release()
    cap1.release()
    cap2.release()
