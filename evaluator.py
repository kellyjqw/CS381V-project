import os
import torch
import pickle
import json
from torchmetrics.classification import (
    BinaryPrecision,
    BinaryRecall,
    BinaryF1Score,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassF1Score,
)
class Evaluator():
    def __init__(self, end_label, out_dir) -> None:
        os.makedirs(out_dir, exist_ok=True)
        # self.tern_prec = MulticlassPrecision(num_classes=3, average="none")
        # self.tern_rec = MulticlassRecall(num_classes=3, average="none")
        self.out_dir = out_dir
        self.tern_f1 = MulticlassF1Score(num_classes=3, average="none")
        self.tern_f1_micro = MulticlassF1Score(num_classes=3, average="micro")

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
            'f1_macro': {'known': [], 'novel': []},
            'f1_micro': {'known': [], 'novel': []}
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
        N = gt_bin.numel()

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
        f1_micro = self.tern_f1_micro(pred, gt)
        k = "novel" if is_novel else "known"
        self.end_state_metrics["f1_0"][k].append(f1[0].item())
        self.end_state_metrics["f1_1"][k].append(f1[1].item())
        self.end_state_metrics["f1_2"][k].append(f1[2].item())
        self.end_state_metrics["f1_macro"][k].append(f1.mean().item())
        self.end_state_metrics["f1_micro"][k].append(f1_micro.item())


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
