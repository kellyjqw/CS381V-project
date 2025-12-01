import torch
from torch import nn
import torch.nn.functional as F 
from torch.nn import LayerNorm
import numpy as np
from typing import Optional
from transformers import CLIPModel, CLIPProcessor


class OTSC_Model(nn.Module):
    def __init__(self, num_classes=4, base_model_name="openai/clip-vit-base-patch16", device: Optional[torch.device] = None, hidden_dim=256):
        super().__init__()
        if device is None:
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = torch.device("mps")
            elif torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")
        self.device = device

        print(f"Loading CLIP backbone: {base_model_name}...")
        self.clip = CLIPModel.from_pretrained(base_model_name)
        self.processor = CLIPProcessor.from_pretrained(base_model_name)

        for param in self.clip.parameters():
            param.requires_grad = False
        self.input_dim = 512 + (512 * 3)
        self.fusion = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.gru = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.progress_head = nn.Linear(hidden_dim, 1)

        self.to(self.device)

    def get_text_features(self, noun, verb):
        prompts = [
            f"a photo of {noun}",
            f"a photo of {noun} being {verb}",
            f"a photo of {noun} {verb}",
        ]

        inputs = self.processor(text=prompts, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            text_features = self.clip.get_text_features(**inputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features.flatten().unsqueeze(0)

    def forward(self, frame, text_embedding_flat, hidden_state=None):
        with torch.no_grad():
            image_features = self.clip.get_image_features(frame)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        cat_features = torch.cat([image_features, text_embedding_flat], dim=1)
        fused = self.fusion(cat_features)
        gru_out, next_hidden = self.gru(fused.unsqueeze(1), hidden_state)
        gru_out_flat = gru_out.squeeze(1)
        logits = self.classifier(gru_out_flat)
        progress = torch.sigmoid(self.progress_head(gru_out_flat))

        return logits, progress, next_hidden
       