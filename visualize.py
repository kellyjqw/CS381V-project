from cgi import print_form
import tqdm
from dataloader import HowToChangeDatasetBatched, custom_collate
import torch
from model import OTSC_Model
from evaluator import Evaluator
# from torch.utils.data import DataLoader
import os
import json
from eval import run_video
from torchvision.io import read_video

class VideoLoader():
    def __init__(self) -> None:
        self.base_path = "data_samples"
        self.videos_path = os.path.join(self.base_path, "clips_cropped")
        csv_path = os.path.join(self.base_path, "test.csv")
        self.df = pd.read_csv(csv_path)   # replace with your path
        def safe_eval(x):
            try:
                return ast.literal_eval(x) if isinstance(x, str) else x
            except:
                return []

        for col in ["initial_state", "transitioning_state", "end_state"]:
            if col in self.df.columns:
                self.df[col] = self.df[col].apply(safe_eval)
            else:
                self.df[col] = [[] for _ in range(len(self.df))]

    def get(self, video_name):
        row = df[df["video_name"] == video_name]
        if not row.empty:
            print(f"ERROR: No such video_name in csv {video_name}")
            return
        row = row.iloc[0]
        init_intervals = row["initial_state"] if isinstance(row["initial_state"], list) else []
        trans_intervals = row["transitioning_state"] if isinstance(row["transitioning_state"], list) else []
        end_intervals = row["end_state"] if isinstance(row["end_state"], list) else []

        video_file = os.path.join(self.video_path, f"{video_name}.mp4")

        if not os.path.exists(video_file):
            print(f"ERROR: No such video_name file {video_name}")
        duration = float(row['duration'])
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
            "is_novel": row["is_novel_osc"]
        }


device = None
if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("using GPU")
else:
    device = torch.device("cpu")
desc_path = "descriptions.json"

# dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=custom_collate)


model = OTSC_Model() 
# ckpt_path = "/lambda/nfs/sc381v-proj"
ckpt_path = "/home/ubuntu/sc381v-proj/CS381V-project/checkpoints2/model_state_3805.pth"
ckpt = torch.load(ckpt_path, map_location=device)
model.load_state_dict(ckpt) 
model.to(device)
model.eval()

import pandas as pd

# 1. Read the CSV into a DataFrame
df = pd.read_csv("your_file.csv")   # replace with your path

# 2. Choose the video_name you want
target_name = "WOQxVNAVwGk_st253.0_dur40.0"   # example

# 3. Get all rows with that video_name
rows = df[df["video_name"] == target_name]
print(rows)
i = 0
torch.set_printoptions(threshold=torch.inf, precision=2, sci_mode=False) 

with torch.no_grad():

    for data in tqdm.tqdm(dataloader):
    # for batch in test_dataset:
        is_novel = data["is_novel"][0]
        preds, gts = run_video(model ,data, device, descriptions_dict)
        pred = torch.tensor(preds[0])
        gt = torch.tensor(gts[0])
        video_name = data["video_name"][0]
        osc=data["osc"][0]
        # print(f"{video_name=}, {osc=}, {is_novel=}, {type(is_novel)}")
        # print(f"{pred=}")
        # print(f"  {gt=}")

        pred_bin = E.bin(pred)
        gt_bin = E.bin(gt)
        E.record_bin_metrics(pred_bin, gt_bin, is_novel)
        E.record_tern_metrics(pred, gt, is_novel)
        E.record_IoU(pred, gt, is_novel)
        E.record_framediff(pred, gt, is_novel)


        # break
        # i+= 1
        # if i == 8:
        #     print(E.end_state_metrics)
        #     break
E.save_result()
