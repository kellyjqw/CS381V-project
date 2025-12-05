import cv2
import torch
from model import OTSC_Model
# from evaluator import Evaluator
# from torch.utils.data import DataLoader
import os
from eval import run_video, load_descriptions
from torchvision.io import read_video
import pandas as pd
import ast
# from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip

target_videos = ["rkkJp1iflIA_st712.0_dur40.0"]
out_dir = "rendered_videos"
in_dir = os.path.join("data_samples", "clips_cropped")
os.makedirs(out_dir, exist_ok=True)

LABEL_MAP = {
    0: "background",
    1: "in_progress",
    2: "finished",
}
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
        row = self.df[self.df["video_name"] == video_name]
        
        if row.empty:
            print(f"ERROR: No such video_name in csv {video_name}")
            return
        row = row.iloc[0]
        init_intervals = row["initial_state"] if isinstance(row["initial_state"], list) else []
        trans_intervals = row["transitioning_state"] if isinstance(row["transitioning_state"], list) else []
        end_intervals = row["end_state"] if isinstance(row["end_state"], list) else []

        video_file = os.path.join(self.videos_path, f"{video_name}.mp4")

        if not os.path.exists(video_file):
            print(f"ERROR: No such video_name file {video_name}")
            return None
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
            "frames": frames.unsqueeze(0),
            "labels": torch.tensor(labels).unsqueeze(0),
            "lengths": torch.tensor([frames.shape[0]]),
            "osc": [row["osc"]],
            "fps": [fps],
            "video_name": [row["video_name"]],
            "video_id": [row["video_id"]],
            "durations": [duration],
            "is_novel": [row["is_novel_osc"]]
        }

   
def label_to_text_and_color(label):
    text = LABEL_MAP.get(label, str(label))
    # Green for "finished" (2), white otherwise
    if label == 2:
        color = (0, 255, 0)  # BGR
    else:
        color = (255, 255, 255)
    return text, color


def add_labels_to_video(
    input_video_path,
    output_video_path,
    preds,
    gts,
    fps=None,       # if None, read from video
    start_sec=0.0,  # you said this will be 0
):
    assert len(preds) == len(gts), "preds and gts must have same length"
    T = len(preds)  # seconds of labels

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file at {input_video_path}")
        return

    # video_fps = cap.get(cv2.CAP_PROP_FPS)
    # if fps is None:
    #     fps = video_fps
    # else:
    #     # If provided fps differs a bit, you can decide which to trust.
    #     pass

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video Info: {width}x{height} @ {fps:.2f} FPS, Total Frames: {total_frames}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # We only need frames covering T seconds starting at start_sec
    start_frame = int(round(start_sec * fps))
    end_frame_exclusive = int(math.ceil((start_sec + T) * fps))
    end_frame_exclusive = min(end_frame_exclusive, total_frames)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    bg_color = (0, 0, 0)  # black background box
    pad = 5
    margin = 10

    frame_idx = start_frame
    while cap.isOpened() and frame_idx < end_frame_exclusive:
        ret, frame = cap.read()
        if not ret:
            break

        # Time (in seconds) since start_sec
        current_time = (frame_idx - start_frame) / fps
        sec_idx = int(current_time)
        if sec_idx >= T:
            sec_idx = T - 1  # safety clamp

        pred_label = preds[sec_idx]
        gt_label   = gts[sec_idx]

        pred_text, pred_color = label_to_text_and_color(pred_label)
        gt_text,   gt_color   = label_to_text_and_color(gt_label)

        pred_draw_text = f"pred: {pred_text}"
        gt_draw_text   = f"gt: {gt_text}"

        # Measure text sizes
        pred_size, _ = cv2.getTextSize(pred_draw_text, font, font_scale, thickness)
        gt_size,   _ = cv2.getTextSize(gt_draw_text,   font, font_scale, thickness)

        overlay = frame.copy()

        # Common baseline y for both captions (bottom of frame)
        y_baseline = height - margin

        # Bottom-left for pred
        px = margin
        py = y_baseline
        cv2.rectangle(
            overlay,
            (px - pad, py - pred_size[1] - pad),
            (px + pred_size[0] + pad, py + pad),
            bg_color,
            -1
        )

        # Bottom-right for gt
        gx = width - margin - gt_size[0]
        gy = y_baseline
        cv2.rectangle(
            overlay,
            (gx - pad, gy - gt_size[1] - pad),
            (gx + gt_size[0] + pad, gy + pad),
            bg_color,
            -1
        )

        # Blend background boxes for readability
        alpha = 0.6
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        # Put the texts
        cv2.putText(frame, pred_draw_text, (px, py), font, font_scale, pred_color, thickness, cv2.LINE_AA)
        cv2.putText(frame, gt_draw_text,   (gx, gy), font, font_scale, gt_color,   thickness, cv2.LINE_AA)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"Saved annotated video to {output_video_path}")



if __name__ == "__main__":
    device = None
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("using GPU")
    else:
        device = torch.device("cpu")
    
    
    # descriptions_dict = load_descriptions()
    descriptions_dict = {}
    model = OTSC_Model() 
    ckpt_path = "/home/ubuntu/sc381v-proj/CS381V-project/checkpoints2/model_state_3805.pth"
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt) 
    model.to(device)
    model.eval()
    VD = VideoLoader()
    
    torch.set_printoptions(threshold=torch.inf, precision=2, sci_mode=False) 
    
    with torch.no_grad():
        for video_name in target_videos:
            data = VD.get(video_name)
            preds, gts = run_video(model ,data, device, descriptions_dict)
            osc = data["osc"][0]
            fps = data["fps"][0]
            osc_dir = os.path.join(out_dir, osc)
            os.makedirs(osc_dir, exist_ok=True)
            input_video_path = os.path.join(in_dir, f"{video_name}.mp4")
            output_video_path = os.path.join(osc_dir, f"{video_name}.mp4")
            render(input_video_path, output_video_path, preds[0], preds[0], fps=fps)
    
          