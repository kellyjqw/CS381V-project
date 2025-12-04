import sys
import os
import ast
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
# import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_video
from transformers import CLIPModel, CLIPProcessor, CLIPConfig
from tqdm import tqdm
from sklearn.metrics import f1_score
from typing import Optional

def visualize_timeline(video_name, gt_list, pred_states, acc, width=80):
    n = len(gt_list)
    if n == 0: return

    step = max(1, n // width)

    def to_char(s):
        if s == 0: return "."
        if s == 1: return "o"
        if s == 2: return "#"
        return "?"

    gt_str = "".join(to_char(x) for x in gt_list[::step])
    pred_str = "".join(to_char(x) for x in pred_states[::step])

    print(f"\nTimeline | {video_name} | Acc: {acc:.2f}")
    print(f"GT   : [{gt_str[:width]}]")
    print(f"Pred : [{pred_str[:width]}]")
    print("-" * 60)
