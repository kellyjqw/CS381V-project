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
class HowToChangeDatasetBatched(Dataset):
    def __init__(self, split='train'):
        self.base_path = "data_samples"
        self.video_path = os.path.join(self.base_path, "clips_cropped")
        self.split = split
        csv_path = os.path.join(self.base_path, f"{self.split}.csv")

        if not os.path.exists(csv_path):
             print(f"Warning: {csv_path} not found.")
             self.annotations = pd.DataFrame(columns=["video_name", "initial_state", "transitioning_state", "end_state", "osc"])
        else:
            self.annotations = pd.read_csv(csv_path)

            def safe_eval(x):
                try:
                    return ast.literal_eval(x) if isinstance(x, str) else x
                except:
                    return []

            for col in ["initial_state", "transitioning_state", "end_state"]:
                if col in self.annotations.columns:
                    self.annotations[col] = self.annotations[col].apply(safe_eval)
                else:
                    self.annotations[col] = [[] for _ in range(len(self.annotations))]

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        row = self.annotations.iloc[idx]
        video_name = row["video_name"]
        osc = row["osc"]
        duration = float(row["duration"])

        init_intervals = row["initial_state"] if isinstance(row["initial_state"], list) else []
        trans_intervals = row["transitioning_state"] if isinstance(row["transitioning_state"], list) else []
        end_intervals = row["end_state"] if isinstance(row["end_state"], list) else []

        video_file = os.path.join(self.video_path, f"{video_name}.mp4")

        if not os.path.exists(video_file):
             return {
                 "valid": False, "osc": osc,
                 "frames": torch.zeros(1), "labels": torch.zeros(1)
             }

        video, _, info = read_video(video_file, pts_unit="sec")
        frames = video.permute(0, 3, 1, 2)
        fps = frames.shape[0] / duration

        labels = []
        num_frames = frames.shape[0]
        for i in range(num_frames):
            t = i / fps
            label = 0
            if any((t >= s) and (t <= e) for (s, e) in trans_intervals):
                label = 1
            if any((t >= s) and (t <= e) for (s, e) in init_intervals):
                label = 1
            if any((t >= s) and (t <= e) for (s, e) in end_intervals):
                label = 2
            labels.append(label)

        labels = torch.tensor(labels, dtype=torch.long)

        return {
            "valid": True,
            "fps": fps,
            "frames": frames,
            "labels": labels,
            "osc": osc,
            "duration": duration,
            "video_name": row["video_name"],
            "video_id": row["video_id"],
            "is_novel": (row["is_novel_osc"] == "True")
        }

def custom_collate(batch):
    batch = [b for b in batch if b["valid"]]
    if len(batch) == 0:
        return None

    max_t = max([b["frames"].shape[0] for b in batch])

    padded_frames = []
    padded_labels = []
    lengths = []
    oscs = []
    fps = []
    durations = []
    # INSERT_YOUR_CODE
    video_names = []
    video_ids = []
    is_novel = []

    for b in batch:
        t = b["frames"].shape[0]
        lengths.append(t)
        oscs.append(b["osc"])
        p_frame = torch.zeros(max_t, 3, 224, 224)
        p_frame[:t] = b["frames"]
        padded_frames.append(p_frame)
        p_label = torch.full((max_t,), -1, dtype=torch.long)
        p_label[:t] = b["labels"]
        padded_labels.append(p_label)
        fps.append(b["fps"])
        durations.append(b["duration"])
        video_names.append(b["video_name"])
        video_ids.append(b["video_id"])
        is_novel.append(b["is_novel"])



    return {
        "frames": torch.stack(padded_frames),
        "labels": torch.stack(padded_labels),
        "lengths": torch.tensor(lengths),
        "osc": oscs, 
        "fps": fps,
        "video_name": video_names,
        "video_id": video_ids,
        "durations":  durations,
        "is_novel": is_novel
    }

class HowToChangeDataLoader(Dataset):
    def __init__(self, clip_path="clips", split='train', test_mode=False):
        if test_mode:
            self.base_path = "data_samples"
            self.video_path = os.path.join(self.base_path, "clips")
        else:
            self.base_path = "HowToChange"
            self.video_path = os.path.join(self.base_path, clip_path)
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
        duration = (row["duration"])
        try:
            verb, noun = osc.split("_", 1)
        except ValueError:
            verb, noun = "change", "object"

        video_file = os.path.join(self.video_path, f"{video_name}.mp4")
        
        if not os.path.exists(video_file):
            print(f"Video {video_file} not found, returning None")
            return None
        
        video, _, info = read_video(video_file, pts_unit="sec")
        frames = video.permute(0, 3, 1, 2) 
        num_frames = frames.shape[0]
        fps = num_frames / duration

        # fps = info.get("video_fps", 30.0)
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
            # "binary_labels": binary_labels,
            "labels": tertiary_labels,
            # "four_labels": four_labels,
            "osc": osc,
            "verb": verb, 
            "noun": noun,
            "duration": duration,
            "video_name": row["video_name"],
            "video_id": row["video_id"]

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


