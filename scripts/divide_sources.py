import os
import constants
from constants import FOLDER_VIDEOS_ORIGINAL
import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm
from pathlib import Path
import datetime
import timeit


def crop_video(
    path: Path, current_num_file: int, tot_num_videos: int, start_time
) -> int:
    if path.is_dir():
        # scan the folder recursively
        for file in path.iterdir():
            current_num_file = crop_video(
                file, current_num_file, tot_num_videos, start_time
            )
        return current_num_file

    current_num_file += 1

    # load video and metadata
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
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

    # create mp4 output
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    # do all the possible versions and only the correct one are going to be kept
    new_width = width // 2
    new_height = height // 2

    suffix_channels = [
        "tl",
        "tr",
        "bl",
        "br",
        "bc",
    ]  # top_left, top_right, bottom_left, bottom_right, bottom_center

    for i in range(5):
        print(f"Channel {i+1}")

        if not cap.isOpened():
            break

        # reset the position to the first frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # create the output file
        relative_pos_index = path.parts.index(FOLDER_VIDEOS_ORIGINAL.name)
        channels_path = constants.FOLDER_VIDEOS.joinpath(
            *path.parts[relative_pos_index + 1 :]
        )
        if not channels_path.parent.exists():
            os.makedirs(channels_path.parent)

        # form path of the specific channel
        channels_path = channels_path.with_name(
            channels_path.name.split(".")[0] + "_" + suffix_channels[i] + ".mp4"
        )

        # avoid processing if it was already processed
        if channels_path.exists():
            continue
        output_video = cv2.VideoWriter(
            str(channels_path), fourcc, fps, (new_width, new_height)
        )
        for _ in tqdm(range(0, int(num_frames))):
            ret, frame = cap.read()
            if not ret:
                break

            x1, x2, y1, y2 = get_boundaries_channel(
                i, width, height, new_width, new_height
            )
            cropped_frame = frame[y1:y2, x1:x2]
            output_video.write(cropped_frame)

        output_video.release()
    cap.release()

    return current_num_file


def get_boundaries_channel(channel, width, height, new_width, new_height) -> tuple:
    if channel == 0:
        # top left
        return (0, new_width, 0, new_height)
    elif channel == 1:
        # top right
        return (new_width, width, 0, new_height)
    elif channel == 2:
        # bottom left
        return (0, new_width, new_height, height)
    elif channel == 3:
        # bottom right
        return (new_width, width, new_height, height)
    elif channel == 4:
        # bottom center
        start_w = new_width // 2
        return (start_w, start_w + new_width, new_height, height)

    # default everything
    return (0, width, 0, height)


total_number_videos = sum(
    len(files) for _, _, files in os.walk(constants.FOLDER_VIDEOS_ORIGINAL)
)
print(f"Total number of videos: {total_number_videos}")
start = timeit.default_timer()
crop_video(
    constants.FOLDER_VIDEOS_ORIGINAL,
    current_num_file=0,
    tot_num_videos=total_number_videos,
    start_time=start,
)
