import constants

import os
import math
from pathlib import Path
import timeit
import json
import collections.abc

import numpy as np
import pandas as pd


def parse_json(
    path: Path,
    current_num_file: int,
    tot_num_files: int,
    start_time,
    initial_parent_folder: str,
) -> int:
    if path.is_dir():
        # scan the folder recursively
        for file in path.iterdir():
            current_num_file = parse_json(
                file,
                current_num_file,
                tot_num_files,
                start_time,
                initial_parent_folder,
            )
        return current_num_file

    # process only json files
    if path.suffix != ".json":
        return current_num_file

    current_num_file += 1

    # output path
    relative_pos_index = path.parts.index(initial_parent_folder)
    parsed_path = constants.FOLDER_ANNOTATIONS_PARSED.joinpath(
        *path.parts[relative_pos_index + 1 :]
    )
    if not parsed_path.parent.exists():
        os.makedirs(parsed_path.parent)
    parsed_path = parsed_path.with_suffix(".csv")

    # features already computed for this video
    if parsed_path.exists():
        return current_num_file

    # parse the data
    with path.open("r") as f:
        data = json.load(f)

        type_annotations = path.name.split(".")[0].split("_")[1]
        if type_annotations == "Turns":
            parseTurns(path, data, parsed_path)
        elif type_annotations == "Focus":
            parseFocus(path, data, parsed_path)

    return current_num_file


def parseTurns(path, data, parsed_path):
    session = path.name.split("_")[0]
    letter_to_pos = constants.MAP_TO_POSITION[session]

    times = []
    positions = []
    for layer_annotations in data["contains"]:
        # each person individually
        if layer_annotations["label"].startswith("TurnUnit") or layer_annotations[
            "label"
        ].startswith("Turn unit"):
            position_pers = letter_to_pos[layer_annotations["label"].split("_")[1]]
            positions.append(position_pers)

            person_output_path = parsed_path.with_name(
                parsed_path.name.split(".")[0] + "_" + position_pers + ".csv"
            )
            times.append([])
            for item in layer_annotations["first"]["items"]:
                target = item["target"]
                # check if an array or only a value
                if isinstance(target, collections.abc.Sequence):
                    time_str = target[0]["selector"]["value"]
                else:
                    time_str = target["selector"]["value"]
                start_time, end_time = time_str.split("=")[1].split(",")
                start_time = float(start_time)
                end_time = float(end_time)
                # account for errors in annotation
                if end_time < start_time:
                    continue

                times[-1].append([start_time, end_time])

            # manage unordered annotations (sort by start_time)
            times[-1] = sorted(times[-1], key=lambda x: x[0], reverse=False)
            times_temp = np.array(times[-1])
            times_temp = pd.DataFrame(times_temp, columns=["Start", "End"])
            # save the data
            times_temp.to_csv(person_output_path, index=False)

    # merging annotations in one timeline
    indexes = [0, 0, 0]
    current_times = [math.inf, math.inf, math.inf]
    talking = [False, False, False]

    past_time_stamp = 0.0
    merged = []

    # unroll times in a long list with start and end
    times_unrolled = []
    for track in times:
        times_unrolled.append([])
        for interval in track:
            times_unrolled[-1].extend(interval)

    # work on times unrolled
    times = times_unrolled

    while True:
        # update considered times
        for k in range(len(times)):
            if indexes[k] == len(times[k]):
                current_times[k] = math.inf
            else:
                current_times[k] = times[k][indexes[k]]

        min_index = np.argmin(current_times)
        min_time = np.min(current_times)

        if min_time == math.inf:
            break

        # add the person talking
        label_talking = ""
        for k in range(len(positions)):
            if talking[k]:
                if label_talking == "":
                    label_talking = positions[k]
                else:
                    label_talking = label_talking + "-" + positions[k]

        # manage no person is talking
        if label_talking == "":
            label_talking = "SILENCE"

        # change from talking to not talking or viceversa
        talking[min_index] = not talking[min_index]

        merged.append([past_time_stamp, min_time, label_talking])
        past_time_stamp = min_time
        indexes[min_index] += 1

    merged = np.array(merged)
    merged = pd.DataFrame(merged, columns=["Start", "End", "Label"])
    # save the data
    merged.to_csv(parsed_path, index=False)


def parseFocus(path, data, parsed_path):
    session = path.name.split("_")[0]
    letter_to_pos = constants.MAP_TO_POSITION[session]

    events = []
    if len(data["contains"]) != 2:
        raise ValueError(
            f"Expected a single layer. Instead {len(data['contains'])} where provided"
        )
    if data["contains"][0]["label"] == "Focus":
        layer_focus = data["contains"][0]
        layer_speaker = data["contains"][1]
    else:
        layer_speaker = data["contains"][0]
        layer_focus = data["contains"][1]

    if layer_focus["label"] != "Focus":
        raise ValueError(
            f"Wrong label. Expected 'Focus', provided {len(layer_focus['label'])}"
        )

    map_speakers = dict()
    for item_speaker in layer_speaker["first"]["items"]:
        target = item_speaker["target"]
        # check if an array or only a value
        if isinstance(target, collections.abc.Sequence):
            time_str = target[0]["selector"]["value"]
        else:
            time_str = target["selector"]["value"]
        start_time, end_time = time_str.split("=")[1].split(",")
        start_time = float(start_time)
        end_time = float(end_time)

        map_speakers[(start_time, end_time)] = item_speaker["body"]["value"].split("+")

    events.append([])
    for index, item in enumerate(layer_focus["first"]["items"]):
        target = item["target"]
        # check if an array or only a value
        if isinstance(target, collections.abc.Sequence):
            time_str = target[0]["selector"]["value"]
        else:
            time_str = target["selector"]["value"]
        start_time, end_time = time_str.split("=")[1].split(",")
        start_time = float(start_time)
        end_time = float(end_time)
        # account for errors in annotation
        if end_time < start_time:
            continue
        type_event = item["body"]["value"]

        speakers = map_speakers.get((start_time, end_time), ["NOTAVAILABLE"])

        label_speakers = ""
        for speaker in speakers:
            if label_speakers != "":
                label_speakers += "-"

            if speaker == "NOTAVAILABLE":
                label_speakers += speaker
            else:
                label_speakers += letter_to_pos[speaker]

        events[-1].append([start_time, end_time, type_event, label_speakers])

    # manage unordered annotations (sort by start_time)
    events[-1] = sorted(events[-1], key=lambda x: x[0], reverse=False)
    # save the data
    events_temp = np.array(events[-1])
    events_temp = pd.DataFrame(
        events_temp, columns=["Start", "End", "Type", "Speakers"]
    )
    events_temp.to_csv(parsed_path, index=False)


total_number_json = 0
for _, _, files in os.walk(constants.FOLDER_ANNOTATIONS_ORIGINAL):
    for file in files:
        if file.split(".")[-1] == "json":
            total_number_json += 1

print(f"Total number of json: {total_number_json}")
start = timeit.default_timer()


parse_json(
    constants.FOLDER_ANNOTATIONS_ORIGINAL,
    current_num_file=0,
    tot_num_files=total_number_json,
    start_time=start,
    initial_parent_folder=constants.FOLDER_ANNOTATIONS_ORIGINAL.name,
)
