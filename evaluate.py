import os

import tqdm
from dataloader import HowToChangeDataLoader
import torch
import torch.nn as nn
# import pytorch_lightning as pl
# import numpy as np
# from data_scripts.evaluator import StatePrec1
from argparse import Namespace
from dataset import HowToChangeFeatDataset

from task import FrameCls
import pickle
import json
from model import OTSC_Model

from torchmetrics.classification import (
    BinaryPrecision,
    BinaryRecall,
    BinaryF1Score,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassF1Score,
)

# vocab = {'background': 0, 'rolling': 1, 'squeezing': 2, 'mashing': 3, 'roasting': 4, 'peeling': 5, 
#         'chopping': 6, 'crushing': 7, 'melting': 8, 'mincing': 9, 'slicing': 10, 'grating': 11, 
#         'sauteing': 12, 'frying': 13, 'blending': 14, 'coating': 15, 'browning': 16, 'grilling': 17, 
#         'shredding': 18, 'whipping': 19, 'zesting': 20}
# sc_list = ['rolling', 'squeezing', 'mashing', 'roasting', 'peeling', 'chopping', 'crushing', 
#             'melting', 'mincing', 'slicing','grating', 'sauteing', 'frying', 'blending', 'coating', 
#             'browning', 'grilling', 'shredding', 'whipping', 'zesting']
# category_num = len(vocab) - 1
# vocab_size = 3 * category_num + 1
# input_dim = 768 
args = Namespace(
    # data args
    ann_dir='./data_files',
    pseudolabel_dir='./videoclip_pseudolabel',
    feat_dir='./data',
    sc_list=["slicing"],
    # model args
    transformer_heads=4,
    transformer_layers=3,
    transformer_dim=512,
    transformer_dropout=0.1,

    log_dir='./logs',
    log_name='debug',
    ckpt="checkpoint/slicing.ckpt",
    use_gt_action=True,
    det=1
)

class Eval():
    def __init__(self, end_label, out_dir) -> None:
        os.makedirs(out_dir, exist_ok=True)
        # self.tern_prec = MulticlassPrecision(num_classes=3, average="none")
        # self.tern_rec = MulticlassRecall(num_classes=3, average="none")
        self.out_dir = out_dir
        self.tern_f1 = MulticlassF1Score(num_classes=3, average="none")
        self.end_label = end_label
        self.end_state_metrics = {
            'first_frame_diff': {'known': [], 'novel': []},
            'iou': {'known': [], 'novel': []},   # only for videos with non-empty end_state intervals
            'total_intersection': {'known': [], 'novel': []},   # to be aggregated over all videos
            'total_union': {'known': [], 'novel': []},   # to be aggregated over all videos
            'accuracy_2': {'known': [], 'novel': []},
            'accuracy_all': {'known': [], 'novel': []},

            'recall': {'known': [], 'novel': []},
            'precision': {'known': [], 'novel': []},
            # 'binary_f1': {'known': [], 'novel': []},
            'f1_0': {'known': [], 'novel': []},
            'f1_1': {'known': [], 'novel': []},
            'f1_2': {'known': [], 'novel': []},
            'f1_all': {'known': [], 'novel': []}
        } 

    def bin(self, x):
        return  (x == self.end_label).long() 

    def tern(self, x: torch.Tensor) -> torch.Tensor:
        # x in {0,1,2,3}
        out = x.clone()
        out[(x == 1) | (x == 2)] = 1
        out[x == 3] = 2
        return out

    def record_bin_metrics(self, pred_bin, gt_bin, is_novel, eps=1e-08):
        N = label.numel()

        tp = ((pred_bin == 1) & (gt_bin == 1)).sum().float()
        fp = ((pred_bin == 1) & (gt_bin == 0)).sum().float()
        fn = ((pred_bin == 0) & (gt_bin == 1)).sum().float()
        tn = ((pred_bin == 0) & (gt_bin == 0)).sum().float()

        precision_pos = tp / (tp + fp + eps)      # positive class = 1
        recall_pos    = tp / (tp + fn + eps)
        f1_pos        = 2 * precision_pos * recall_pos / (precision_pos + recall_pos + eps)
        acc           = (tp + tn) / N
        k = "novel" if is_novel else "known"
        self.end_state_metrics["precision"][k].append(precision_pos.item())
        self.end_state_metrics["recall"][k].append(recall_pos.item())
        # self.end_state_metrics["binary_f1"][k].append(f1_pos.item())
        self.end_state_metrics["accuracy_2"][k].append(acc.item())
    
    def record_tern_metrics(self, pred, gt, is_novel):
        f1 = self.tern_f1(pred, gt)
        k = "novel" if is_novel else "known"
        self.end_state_metrics["f1_0"][k].append(f1[0].item())
        self.end_state_metrics["f1_1"][k].append(f1[1].item())
        self.end_state_metrics["f1_2"][k].append(f1[2].item())
        self.end_state_metrics["f1_all"][k].append(f1.mean().item())

        num_correct = (pred == gt).sum().item()
        num_total = gt.numel()
        accuracy = num_correct / num_total
        self.end_state_metrics["accuracy_all"][k].append(accuracy)


    # def record_accuracy(self, pred, gt, is_novel):
    #     """
    #     Record framewise accuracy for predictions compared to ground truth.
    #     """
    #     acc = (pred == gt).float().mean().item()
    #     k = "novel" if is_novel else "known"
    #     self.end_state_metrics["accuracy_all"][k].append(acc)


    def record_IoU(self, pred, gt, is_novel):
        """
        Compute the Intersection over Union (IoU) between predicted and ground truth end state regions.
        Treats all indices where value == 3 as the region.
        """
        k = "novel" if is_novel else "known"

        pred_region = (pred == self.end_label)
        gt_region = (gt == self.end_label)

        intersection = (pred_region & gt_region).sum().item()
        union = (pred_region | gt_region).sum().item()

        self.end_state_metrics["total_intersection"][k].append(intersection)
        self.end_state_metrics["total_union"][k].append(union)

        if gt_region.any(): 
            iou = intersection / union 
            self.end_state_metrics["iou"][k].append(iou)

    def record_framediff(self, pred, gt, is_novel):  # pred, gt: (T,) tensors
        """
        Record the difference in the first appearing frame index of '3' class between pred and gt.
        If '3' never occurs, use -1 as the index.
        """
        T = gt.numel()

        pred_idx = (pred == self.end_label).nonzero(as_tuple=True)[0]
        gt_idx = (gt == self.end_label).nonzero(as_tuple=True)[0]
        
        pred_first_idx = pred_idx[0].item() if len(pred_idx) > 0 else T
        gt_first_idx = gt_idx[0].item() if len(gt_idx) > 0 else T

        diff = abs(pred_first_idx - gt_first_idx)
        k = "novel" if is_novel else "known"
        self.end_state_metrics["first_frame_diff"][k].append(diff)

    def save_result(self):
        pkl_path = os.path.join(self.out_dir, "results.pkl")
        json_path = os.path.join(self.out_dir, "results.json")

        # Save the metrics dictionary as a pickle file
        with open(pkl_path, "wb") as f:
            pickle.dump(self.end_state_metrics, f)

        with open(json_path, "w") as f:
            json.dump(self.end_state_metrics, f, indent=2)


def predict(model, frames):
    pred = model()
    # post process, if any
    return pred




test_dataset = HowToChangeDataLoader(split="test")
E = Eval(end_label=2, out_dir="results")
model = OTSC_Model()  # make sure you have the right version of model in model.py!
model.eval()
i = 0
torch.set_printoptions(threshold=torch.inf, precision=2, sci_mode=False) 

with torch.no_grad():

    for data in tqdm.tqdm(test_dataset):
    # for batch in test_dataset:

    # in progress
        preds_ps = []
        labels_ps = []

        labels = data["tertiary_labels"]
        frames = data["frames"].float()
        length = data["length"]
        osc_name = data["osc"]
        fps = data["fps"]
            
        Max_T = frames.shape[0]
        pbar = tqdm(range(Max_T), leave=False)

        text_flat, text_raw = model.get_text_features_batch(osc_name, descriptions_dict)

        hidden_state = None
        progress_buffer = []

        total_loss = 0
        pbar = tqdm(range(Max_T), desc=f"", leave=False)
        for t in pbar:
            active_mask = (length > t).float().unsqueeze(1)
            if active_mask.sum() == 0: break

            logits, progress, hidden_state, img_feat = model(frames[t], text_flat, hidden_state)
            prior_logits = torch.matmul(img_feat.unsqueeze(1), text_raw.transpose(1, 2)).squeeze(1)
            progress_buffer.append(progress.detach())

            if len(progress_buffer) > K_WINDOW:
                progress_buffer.pop(0)

            pred = torch.argmax(logits, dim=-1).cpu()

        E.record_bin_metrics(pred_bin, gt_bin, is_novel)
        E.record_tern_metrics(pred_3, gt_3, is_novel)
        E.record_IoU(pred_4, gt_4, is_novel)
        E.record_framediff(pred_4, gt_4, is_novel)

        # break
        # i+= 1
        # if i % 300 == 0:
        #     E.save_result()
        #     break
E.save_result()
print(f"num samples: {i}")
