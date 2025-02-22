import constants
import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
import torchvision.transforms as T
from torchvision.io import read_video
from torchvision.models.optical_flow import (
    raft_large,
    raft_small,
    Raft_Small_Weights,
    Raft_Large_Weights,
)
from torchvision.utils import flow_to_image
import tqdm

plt.rcParams["savefig.bbox"] = "tight"


def plot(imgs, **imshow_kwargs):
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0])
    _, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
    for row_idx, row in enumerate(imgs):
        for col_idx, img in enumerate(row):
            ax = axs[row_idx, col_idx]
            img = F.to_pil_image(img.to("cpu"))
            ax.imshow(np.asarray(img), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    plt.tight_layout()
    plt.show()


video = str(constants.FOLDER_VIDEOS) + "/video_test.mp4"

video_frames, _, metadata = read_video(video)
video_fps = metadata["video_fps"]
print(f"The video runs at {video_fps:.2f} fps")
video_frames = video_frames.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)


def preprocess(batch):
    transforms = T.Compose(
        [
            T.ConvertImageDtype(torch.float32),
            T.Normalize(
                mean=0.5, std=0.5
            ),  # map [0, 1] into [-1, 1] substracting the mean and dividing by the std
        ]
    )
    batch = transforms(batch)
    return batch


device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device: " + device)

video_frames = preprocess(video_frames).to(device)


if constants.USE_RAFT_LARGE:
    model = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=True).to(device)
else:
    model = raft_small(weights=Raft_Small_Weights.DEFAULT, progress=True).to(device)
model = model.eval()

for index_frame in tqdm.tqdm(range(0, len(video_frames), 2)):
    if index_frame + 1 >= len(video_frames):
        break
    img1_batch = torch.stack([video_frames[index_frame]])
    img2_batch = torch.stack(
        [video_frames[index_frame + 1]]
    )

    # flows of each iteration
    list_of_flows = model(img1_batch.to(device), img2_batch.to(device))
    predicted_flows = list_of_flows[-1]  # shape = (N, 2, H, W)
    flow_imgs = flow_to_image(predicted_flows)

    # Transfrom from [-1, 1] to [0, 1]
    # img1_batch = [(img1 + 1) / 2 for img1 in img1_batch]

    # grid = [[img1, flow_img] for (img1, flow_img) in zip(img1_batch, flow_imgs)]
    # for element in grid:
    #     plot(element)
