import pickle
import numpy as np
import pandas as pd
import sys
from pathlib import Path

def summarize_metrics(pkl_path, csv_path):
    # Load dictionary from pkl
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    rows = []

    # Handle all metrics except total_intersection / total_union
    for metric_name, values in data.items():
        if metric_name in ("total_intersection", "total_union"):
            continue

        known_vals = np.array(values["known"], dtype=float)
        novel_vals = np.array(values["novel"], dtype=float)
        all_vals = np.concatenate([known_vals, novel_vals])

        rows.append(
            {
                "metric": metric_name,
                "known": known_vals.mean(),
                "novel": novel_vals.mean(),
                "all": all_vals.mean(),
            }
        )

    # Replace total_intersection / total_union with total_iou metric
    ti = data["total_intersection"]
    tu = data["total_union"]

    # IoU for known
    known_iou = float(np.sum(ti["known"])) / float(np.sum(tu["known"]))
    # # IoU for novel
    novel_iou = float(np.sum(ti["novel"])) / float(np.sum(tu["novel"]))
    novel_iou = 0
    # IoU for all
    all_iou = float(np.sum(ti["known"]) + np.sum(ti["novel"])) / float(
        np.sum(tu["known"]) + np.sum(tu["novel"])
    )

    rows.append(
        {
            "metric": "total_iou",
            "known": known_iou,
            "novel": novel_iou,
            "all": all_iou,
        }
    )

    # Build DataFrame and save
    df = pd.DataFrame(rows).set_index("metric")
    df.to_csv(csv_path)
    print(f"Saved summary to {csv_path}")

summarize_metrics(pkl_path="results/results.pkl", csv_path="results/result_summary.csv")