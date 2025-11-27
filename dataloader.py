''' documentation see README'''

import ast
import torch
from torchvision.io import read_video
import pandas as pd
from torch.utils.data import Dataset
import os

class HowToChangeDataLoader(Dataset):
    def __init__(self, clip_path, split='train', test_mode=False):
        if test_mode:
            self.base_path = "data_samples"
            self.video_path = os.path.join(self.base_path, "clips")
        else:
            self.base_path = "HowToChange"
            self.video_path = os.path.join(self.base_path, clip_path)
        self.split = split
        self.annotations = pd.read_csv(os.path.join(self.base_path, f"{self.split}.csv"))
        self.annotations["end_intervals"] = self.annotations["end_state"].apply(
            lambda s: ast.literal_eval(s)
        )
    def __len__(self):
        return len(self.annotations)
        
    def __getitem__(self, idx):
        row = self.annotations.iloc[idx]
        video_name = row["video_name"]
        end_intervals = row["end_intervals"]  # already list of [start, end]
        print("end_intervals", end_intervals)
        osc = row["osc"]
        verb, noun = osc.split("_", 1)  

        video_file = os.path.join(self.video_path, f"{video_name}.mp4")
        # video: (T, H, W, C), audio: ignored, info: dict with fps
        video, _, info = read_video(video_file, pts_unit="sec")
        # Convert to (T, C, H, W), uint8
        frames = video.permute(0, 3, 1, 2) 
        num_frames = frames.shape[0]

        fps = info.get("video_fps", None)
        if fps is None or fps == 0:
            duration = float(row["duration"])
            fps = num_frames / duration

        labels = []
        for i in range(num_frames):
            t = i / fps  # seconds from clip start
            is_end = any((t >= s) and (t <= e) for (s, e) in end_intervals)
            labels.append(1 if is_end else 0)
        labels = torch.tensor(labels)

        return {
            "fps": fps, 
            "frames": frames,   # Tensor: [T, C, H, W]
            "labels": labels,   # Tensor: [T]
            "osc": osc,
            "verb": verb, 
            "noun": noun
        }

if __name__ == "__main__":
    torch.set_printoptions(threshold=torch.inf) 
    dataloader = HowToChangeDataLoader(test_mode=True)
    for sample in dataloader:
        frames = sample["frames"]          # [T, C, H, W]
        T, C, H, W = frames.shape
        print("Single sample shape:", frames.shape)
        print("Resolution (H, W):", H, W)
        print('fps', sample['fps'])
        print('osc', sample['osc'])
        print('verb', sample['verb'])
        print('noun', sample['noun'])
        # print(sample['labels'])


