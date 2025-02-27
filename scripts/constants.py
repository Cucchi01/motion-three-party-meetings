from pathlib import Path

CURRENT_PATH = Path(__file__)
CURRENT_PATH = CURRENT_PATH.resolve()

# move the root folder
while CURRENT_PATH.absolute().parts[-1] != "motion-three-party-meetings":
    CURRENT_PATH = CURRENT_PATH.parent

FOLDER_VIDEOS = Path(*CURRENT_PATH.absolute().parts, "data", "video", "channels")
FOLDER_VIDEOS_ORIGINAL = Path(
    *CURRENT_PATH.absolute().parts, "data", "MEETdata-group13", "videos"
)
FOLDER_FEATURES = Path(*CURRENT_PATH.absolute().parts, "data", "features")
FOLDER_FEATURES_MOTION = FOLDER_FEATURES.joinpath("motions")

SHAPE_PREDICTOR_PATH = Path(
    *CURRENT_PATH.absolute().parts, "weights", "shape_predictor_68_face_landmarks.dat"
)

USE_RAFT_LARGE = False


class OpticalFlowConstansts:
    PADDING_BOX_BOTTOM_FACE = 10
    VISUALIZE = False
    VISUALIZE_EVERY_NUM_FRAMES = 30
    # do not consider all the frames to lower the amount of computations
    KEEP_EVERY_NUM_FRAMES = 10
    DOWNSAMPLE_RATE = 2

    WINDOW_LENGTH = 2.0
