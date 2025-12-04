
import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor, CLIPConfig
from typing import Optional

class OTSC_Model(nn.Module):
    def __init__(self, model_id="zer0int/LongCLIP-GmP-ViT-L-14", device: Optional[torch.device] = None, hidden_dim=256):
        super().__init__()
        if device is None:
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = torch.device("mps")
            elif torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")
        self.device = device

        print(f"Loading LongCLIP backbone: {model_id}...")
        config = CLIPConfig.from_pretrained(model_id)
        config.text_config.max_position_embeddings = 248

        self.clip = CLIPModel.from_pretrained(model_id, config=config)
        self.processor = CLIPProcessor.from_pretrained(model_id, config=config)

        for param in self.clip.parameters():
            param.requires_grad = False

        self.proj_dim = self.clip.config.projection_dim
        self.input_dim = self.proj_dim + (self.proj_dim * 3)

        self.fusion = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.gru = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True)

        self.classifier = nn.Linear(hidden_dim, 3)
        self.progress_head = nn.Linear(hidden_dim, 1)

        self.to(self.device)

    def get_text_features_batch(self, osc_names_list, descriptions_dict):
        batch_prompts = []

        for osc in osc_names_list:
            default_desc = {
                "progress_description": f"Start and in-progress state of {osc.replace('_', ' ')}.",
                "finished_description": f"Finished state of {osc.replace('_', ' ')}."
            }
            desc = descriptions_dict.get(osc, default_desc)
            batch_prompts.append(f"Initial state: {desc.get('progress_description', '')}")
            batch_prompts.append(f"Transitioning: {desc.get('progress_description', '')}")
            batch_prompts.append(f"Final state: {desc.get('finished_description', '')}")

        inputs = self.processor(
            text=batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=248
        ).to(self.device)

        with torch.no_grad():
            all_feats = self.clip.get_text_features(**inputs)
            all_feats = all_feats / all_feats.norm(dim=-1, keepdim=True)
        B = len(osc_names_list)
        Dim = all_feats.shape[-1]

        raw_feat = all_feats.view(B, 3, Dim)
        flat_feat = raw_feat.reshape(B, -1)

        return flat_feat, raw_feat

    def forward(self, frame_batch, text_embedding_batch, hidden_state=None):
        with torch.no_grad():
            image_features = self.clip.get_image_features(frame_batch)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        cat_features = torch.cat([image_features, text_embedding_batch], dim=1)
        fused = self.fusion(cat_features)

        gru_out, next_hidden = self.gru(fused.unsqueeze(1), hidden_state)
        gru_out_flat = gru_out.squeeze(1)

        logits = self.classifier(gru_out_flat)
        progress = torch.sigmoid(self.progress_head(gru_out_flat))

        return logits, progress, next_hidden, image_features