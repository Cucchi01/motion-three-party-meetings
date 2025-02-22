from pathlib import Path

CURRENT_PATH = Path(__file__)
CURRENT_PATH = CURRENT_PATH.resolve()

# move the root folder
while CURRENT_PATH.absolute().parts[-1] != "motion-three-party-meetings":
    CURRENT_PATH = CURRENT_PATH.parent

FOLDER_VIDEOS = Path(*CURRENT_PATH.absolute().parts, "data", "videos")
FOLDER_FEATURES = Path(*CURRENT_PATH.absolute().parts, "data", "features")

SHAPE_PREDICTOR_PATH = Path(
    *CURRENT_PATH.absolute().parts, "weights", "shape_predictor_68_face_landmarks.dat"
)

USE_RAFT_LARGE = False
