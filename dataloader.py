import torch
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
import os

class HowToChangeDataLoader(Dataset):
    def __init__(self,split='train'):
        pass
    def __len__(self):
        pass
    def __getitem__(self, idx):
        output_dict = {
            'video_features': video_features.squeeze(1),
            'video_padding_mask': torch.zeros(video_features.size(0), dtype=torch.bool),
            'narration_features': padded_narration_features.squeeze(1),
            'narration_padding_mask': narration_padding_mask,
            'starts': padded_starts.squeeze(1),
            'ends': padded_ends.squeeze(1),
            'metadata' : metadata
        }

        output_dict['mean'] = (output_dict['starts'] + output_dict['ends']) / 2
        output_dict['duration'] = torch.abs(output_dict['ends']-output_dict['starts'])

        return output_dict
