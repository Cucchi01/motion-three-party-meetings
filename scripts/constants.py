from pathlib import Path

CURRENT_PATH = Path(__file__)
CURRENT_PATH = CURRENT_PATH.resolve()

# move the root folder
while CURRENT_PATH.absolute().parts[-1] != "motion-three-party-meetings":
    CURRENT_PATH = CURRENT_PATH.parent

FOLDER_VIDEOS = Path(*CURRENT_PATH.absolute().parts, "data", "video", "channels")
FOLDER_VIDEOS_TEST = Path(
    *CURRENT_PATH.absolute().parts, "data", "video", "videos_test"
)
FOLDER_VIDEOS_ORIGINAL = Path(
    *CURRENT_PATH.absolute().parts, "data", "MEETdata-group13", "videos"
)
FOLDER_FEATURES = Path(*CURRENT_PATH.absolute().parts, "data", "features")
FOLDER_FEATURES_MOTION = FOLDER_FEATURES.joinpath("motions")
FOLDER_ANNOTATIONS_ORIGINAL = Path(
    *CURRENT_PATH.absolute().parts, "data", "annotations"
)
FOLDER_ANNOTATIONS_PARSED = FOLDER_ANNOTATIONS_ORIGINAL.joinpath("parsed")


SHAPE_PREDICTOR_PATH = Path(
    *CURRENT_PATH.absolute().parts, "weights", "shape_predictor_68_face_landmarks.dat"
)


MAP_TO_POSITION = {
    "SESS001": {"C": "bl", "A": "tl", "B": "br"},
    "SESS002": {"B": "bl_tr", "C": "br_bc", "A": "tl"},
    "SESS003": {"A": "tl", "B": "tr", "C": "bc"},
    "SESS004": {"C": "bc", "B": "tr", "A": "tl"},
    "SESS005": {"A": "tl", "B": "bl", "C": "br"},
    "SESS006": {"C": "bc", "B": "tr", "A": "tl"},
    "SESS009": {"C": "bc", "A": "tl", "B": "tr"},
}


class OpticalFlowConstansts:
    USE_RAFT_LARGE = True

    PADDING_BOX_BOTTOM_FACE = 60
    VISUALIZE = False
    VIDEO_TEST = False
    # do not consider all the frames to lower the amount of computations
    KEEP_EVERY_NUM_FRAMES = 10
    VISUALIZE_EVERY_NUM_FRAMES = 2
    DOWNSAMPLE_RATE = 1

    AVERAGE_BY_WINDOW = False
    WINDOW_LENGTH = 2.0
