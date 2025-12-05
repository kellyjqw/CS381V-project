from cgi import print_form
import tqdm
from dataloader import HowToChangeDataLoader, custom_collate
import torch
from model import OTSC_Model
from torch.utils.data import DataLoader
import os
import json
import pandas as pd
import time

total_time = 0.0
total_frames = 0
def run_video(model, batch_data, device, descriptions_dict):
    global total_time, total_frames  # <-- so we can update the globals

    frames_all = batch_data["frames"].float().to(device)
    labels_all = batch_data["labels"].to(device)
    lengths = batch_data["lengths"].to(device)
    osc_names = batch_data["osc"]
    fps = batch_data["fps"][0]
    duration = batch_data["durations"]

    B, Max_T = frames_all.shape[:2]
    Max_D = int(max(duration))
    text_flat, text_raw = model.get_text_features_batch(osc_names, descriptions_dict)

    hidden_state = None
    progress_buffer = []

    batch_preds = [[] for _ in range(B)]
    batch_gts = [[] for _ in range(B)]

    start = time.time()
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

    end = time.time()
    total_time += end - start
    total_frames += len(batch_preds[0])
    hidden_state = hidden_state.detach()
    return batch_preds, batch_gts

# def IoU(pred, gt, end_label):
#     """
#     Compute the Intersection over Union (IoU) between predicted and ground truth end state regions.
#     Treats all indices where value == 3 as the region.
#     """
#     global total_iou
#     global count
#     pred_region = (pred == end_label)
#     gt_region = (gt == end_label)

#     intersection = (pred_region & gt_region).sum().item()
#     union = (pred_region | gt_region).sum().item()

#     if gt_region.any(): 
#         iou = intersection / union 
#         total_iou += iou
#         count += 1
#     else:
#         iou = 0.0
#     print(f"{iou=}")
#     return iou
        
def load_descriptions():
    desc_path = "/home/ubuntu/sc381v-proj/data_samples/multiclass_descriptions.json"
    if os.path.exists(desc_path):
        with open(desc_path, "r") as f:
            descriptions_dict = json.load(f)
        print(f"Loaded {len(descriptions_dict)} descriptions.")
    else:
        descriptions_dict = {}
        print(f"NOT LOADING DESCRIPTIONS")
    return descriptions_dict

if __name__ == "__main__":
      
    device = None
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("using GPU")
    else:
        device = torch.device("cpu")
    # descriptions_dict = {}
    # print(f"NOT LOADING DESCRIPTIONS")

    descriptions_dict = load_descriptions()
    dataset = HowToChangeDataLoader(split="test")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=custom_collate)
    model = OTSC_Model() 
    # ckpt_path = "/lambda/nfs/sc381v-proj"
    ckpt_path = "/home/ubuntu/sc381v-proj/CS381V-project/checkpoints6/final_model.pt"
    # ckpt_path = "CS381V-project/checkpoints2/model_state_3805.pth"
    # ckpt_path = "model_state_3805_v2.pth"

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state']) 
    model.to(device)
    model.eval()
    i = 0
    rows = []
    torch.set_printoptions(threshold=torch.inf, precision=2, sci_mode=False) 
    
    with torch.no_grad():
    
        for data in tqdm.tqdm(dataloader):
        # for data in dataloader:
            # is_novel = data["is_novel"][0]
            preds, gts = run_video(model ,data, device, descriptions_dict)
            pred = preds[0]
            gt = gts[0]
            video_name = data["video_names"][0]
            # osc=data["osc"][0]
            # iou = IoU(torch.tensor(pred), torch.tensor(gt), 2)
            
    
            rows.append({
                "video_name": video_name,
                "pred": " ".join(map(str, pred)),
                "gt": " ".join(map(str, gt)),
                # "iou": iou,
                # "osc": osc
            })

# print("IoU result:", total_iou / count)
df = pd.DataFrame(rows)
df.to_csv("LLM_inference_result_test.csv", index=False)
print(f"Average time per frame: {total_time / total_frames}s")

# reading in:
# df = pd.read_csv("baseline_inference_result.csv")

# df["pred_list"] = df["pred"].apply(lambda s: [int(x) for x in s.split()])
# df["gt_list"]   = df["gt"].apply(lambda s: [int(x) for x in s.split()])
