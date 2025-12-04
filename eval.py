import tqdm
from dataloader import HowToChangeDatasetBatched
import torch
from model import OTSC_Model
from evaluator import Evaluator
from torch.utils.data import DataLoader
import os
import json

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

def run_video(model, batch_data, device, descriptions_dict):
    frames_all = batch_data["frames"].float().to(device)
    labels_all = batch_data["labels"].to(device)
    lengths = batch_data["lengths"].to(device)
    osc_names = batch_data["osc"]
    fps = batch_data["fps"]
    duration = batch_data["durations"]

    B, Max_T = frames_all.shape[:2]
    Max_D = int(max(duration))
    text_flat, text_raw = model.get_text_features_batch(osc_names, descriptions_dict)

    hidden_state = None
    progress_buffer = []

    batch_preds = [[] for _ in range(B)]
    batch_gts = [[] for _ in range(B)]

    for s in range(Max_D):
        t = min(int(fps*s), Max_T)
        active_mask = (lengths > t).float().unsqueeze(1)
        if active_mask.sum() == 0: break

        current_frames = frames_all[:, t]
        current_labels = labels_all[:, t]

        logits, progress, hidden_state, img_feat = model(current_frames, text_flat, hidden_state)
        prior_logits = torch.matmul(img_feat.unsqueeze(1), text_raw.transpose(1, 2)).squeeze(1)

        current_preds = torch.argmax(logits, dim=-1).cpu().numpy()
        for i in range(B):
            if t < lengths[i]:
                batch_preds[i].append(current_preds[i])
                batch_gts[i].append(labels_all[i, t].item())
    hidden_state = hidden_state.detach()
    return batch_preds, batch_gts

device = None
if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("using GPU")
else:
    device = torch.device("cpu")
desc_path = "descriptions.json"
if os.path.exists(desc_path):
    with open(desc_path, "r") as f:
        descriptions_dict = json.load(f)
    print(f"Loaded {len(descriptions_dict)} descriptions.")
else:
    descriptions_dict = {}
dataset = HowToChangeDatasetBatched(split="test")
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=custom_collate)
E = Evaluator(end_label=2, out_dir="results")
model = OTSC_Model()  
model.eval()
i = 0
torch.set_printoptions(threshold=torch.inf, precision=2, sci_mode=False) 

with torch.no_grad():

    for data in tqdm.tqdm(dataloader):
    # for batch in test_dataset:
        is_novel = data["is_novel"][0]
        preds, gts = run_video(model ,data, device, descriptions_dict)
        pred = preds[0]
        gt = gts[0]
        pred_bin = E.bin(pred)
        gt_bin = E.bin(gt)
        E.record_bin_metrics(pred_bin, gt_bin, is_novel)
        E.record_tern_metrics(pred, gt, is_novel)
        E.record_IoU(pred, gt, is_novel)
        E.record_framediff(pred, gt, is_novel)

        # break
        # i+= 1
        # if i % 300 == 0:
        #     E.save_result()
        #     break
E.save_result()
