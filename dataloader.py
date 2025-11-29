''' usage: 
dataset = HowToChangeDataLoader()
dataloader = DataLoader(dataset, batch_size=x, shuffle=True, collate_fn=collate_fn, num_workers=0)
for epoch in range(epochs):
    for batch in dataloader:
        batch["four_labels"][b][i]  # the i_th frame label of the b_th video clip in the batch


no batching: 
dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
for epoch in range(epochs):
    for batch in dataloader:
        batch["four_labels"][i]  # no extra batch dimension
'''

import ast
import torch
from torchvision.io import read_video
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import os


class HowToChangeDataLoader(Dataset):
    def __init__(self, split='train', test_mode=False):
        if not test_mode:
            print("Note: Currently utilizing test_mode for online simulation.")
        self.base_path = "data_samples"
        self.video_path = os.path.join(self.base_path, "clips")
        self.split = split
        csv_path = os.path.join(self.base_path, f"{self.split}.csv")
        if not os.path.exists(csv_path):
             print(f"Warning: {csv_path} not found. Using dummy data structure.")
             self.annotations = pd.DataFrame(columns=["video_name", "end_intervals", "osc", "duration"])
        else:
            self.annotations = pd.read_csv(csv_path)
            self.annotations["end_intervals"] = self.annotations["end_state"].apply(
              lambda s: ast.literal_eval(s) if isinstance(s, str) else s
            )
            self.annotations["transitioning_intervals"] = self.annotations["transitioning_state"].apply(
              lambda s: ast.literal_eval(s) if isinstance(s, str) else s
            )
            self.annotations["initial_intervals"] = self.annotations["initial_state"].apply(
              lambda s: ast.literal_eval(s) if isinstance(s, str) else s
            )
    def __len__(self):
        return len(self.annotations)
        
    def __getitem__(self, idx):
        row = self.annotations.iloc[idx]
        video_name = row["video_name"]
        end_intervals = row.get("end_intervals", [])
        transition_intervals = row.get("transitioning_intervals", [])
        initial_intervals = row.get("initial_intervals", [])
        osc = row["osc"]
        try:
            verb, noun = osc.split("_", 1)
        except ValueError:
            verb, noun = "change", "object"

        video_file = os.path.join(self.video_path, f"{video_name}.mp4")
        
        if not os.path.exists(video_file):
             print(f"Video {video_file} not found.")
             return {
                "fps": 30.0,
                "frames": torch.zeros(30, 3, 224, 224), 
                "labels": torch.zeros(30),
                "osc": osc, "verb": verb, "noun": noun
             }
        
        video, _, info = read_video(video_file, pts_unit="sec")
        frames = video.permute(0, 3, 1, 2) 
        num_frames = frames.shape[0]

        fps = info.get("video_fps", 30.0)
        binary_labels = []
        for i in range(num_frames):
            t = i / fps
            is_end = any((t >= s) and (t <= e) for (s, e) in end_intervals)
            binary_labels.append(1 if is_end else 0)
        binary_labels = torch.tensor(binary_labels)
        binary_labels = []
        tertiary_labels = []
        four_labels = []
        for i in range(num_frames):
            t = i / fps
            if any((t >= s) and (t <= e) for (s, e) in end_intervals):
                binary_labels.append(1)
                tertiary_labels.append(2)
                four_labels.append(3)
            elif any((t >= s) and (t <= e) for (s, e) in transition_intervals):
                binary_labels.append(0)
                tertiary_labels.append(1)
                four_labels.append(2)
            elif any((t >= s) and (t <= e) for (s, e) in initial_intervals):
                binary_labels.append(0)
                tertiary_labels.append(1)
                four_labels.append(1)
            else:
                binary_labels.append(0)
                tertiary_labels.append(0)
                four_labels.append(0)
        binary_labels = torch.tensor(binary_labels)
        tertiary_labels = torch.tensor(tertiary_labels)
        four_labels = torch.tensor(four_labels)

        return {
            "fps": fps, 
            "frames": frames,
            "binary_labels": binary_labels,
            "tertiary_labels": tertiary_labels,
            "four_labels": four_labels,
            "osc": osc,
            "verb": verb, 
            "noun": noun
        }
    
def collate_fn(batch):
    # batch is a list of dicts from __getitem__
    fps = torch.tensor([b["fps"] for b in batch], dtype=torch.float32)

    frames = [b["frames"] for b in batch]               # list of (T_i, 3, H, W)
    binary_labels = [b["binary_labels"] for b in batch] # list of (T_i,)

    binary_labels = [b["binary_labels"] for b in batch]               # list of (T_i,)
    tertiary_labels = [b["tertiary_labels"] for b in batch]               # list of (T_i,)
    four_labels = [b["four_labels"] for b in batch]               # list of (T_i,)

    osc = [b["osc"] for b in batch]
    verb = [b["verb"] for b in batch]
    noun = [b["noun"] for b in batch]

    return {
        "fps": fps,
        "frames": frames,
        "binary_labels": binary_labels,
        "tertiary_labels": tertiary_labels,
        "four_labels": four_labels,
        "osc": osc,
        "verb": verb,
        "noun": noun,
    }

if __name__ == "__main__":
    torch.set_printoptions(threshold=torch.inf) 

    dataset = HowToChangeDataLoader(test_mode=True)
    dataloader = DataLoader(dataset, batch_size=3, shuffle=True, collate_fn=collate_fn)
    for e in range(3):
        print(f"EPOCH {e}")
        for batch in dataloader:
            # print(batch)
            frames = batch["frames"][1]          # [T, C, H, W]
            T, C, H, W = frames.shape
            print("Single sample shape:", frames.shape)
            print("Resolution (H, W):", H, W)
            print('fps', batch['fps'][1])
            print('osc', batch['osc'][1])
            print('verb', batch['verb'][1])
            print('noun', batch['noun'][1])


