import os
import torch
import torch.nn as nn
import numpy as np
import h5py
from torch.utils.data import Dataset, DataLoader


class TextOnlyMLP(nn.Module):
    def __init__(self, text_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class AudioOnlyMLP(nn.Module):
    def __init__(self, audio_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(audio_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class MultiModalMLP(nn.Module):
    def __init__(self, text_dim, audio_dim, hidden_dim, output_dim):
        super().__init__()
        self.text_mlp = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.audio_mlp = nn.Sequential(
            nn.Linear(audio_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.fusion_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, text_input, audio_input):
        text_feat = self.text_mlp(text_input)
        audio_feat = self.audio_mlp(audio_input)
        fused = torch.cat((text_feat, audio_feat), dim=1)
        return self.fusion_mlp(fused)
   

def load_model(mode, device):
    TEXT_DIM = AUDIO_DIM = 768  # adjust if needed
    HIDDEN_DIM = 128
    OUTPUT_DIM = 3

    if mode == "text":
        model = TextOnlyMLP(TEXT_DIM, HIDDEN_DIM, OUTPUT_DIM)
        model_path = "static/assets/models/text_only_class_model_normalized_t70-3.pth"
    elif mode == "audio":
        model = AudioOnlyMLP(AUDIO_DIM, HIDDEN_DIM, OUTPUT_DIM)
        model_path = "static/assets/models/audio_only_class_model_normalized_t70-3.pth"
    else:
        model = MultiModalMLP(TEXT_DIM, AUDIO_DIM, HIDDEN_DIM, OUTPUT_DIM)
        model_path = "static/assets/models/streamer_class_model_normalized_t70-3.pth"

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def get_preds_class(mode, text="False", audio="False"):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MODE = mode  # change as needed: "text", "audio", or "both"
    model = load_model(mode=MODE, device=DEVICE)
    if MODE in ["text", "both"]:
        mean_text = torch.tensor(np.load("static/assets/norm_params/text_class_mean.npy"), dtype=torch.float32).to(DEVICE)
        std_text = torch.tensor(np.load("static/assets/norm_params/text_class_std.npy"), dtype=torch.float32).to(DEVICE)
    if MODE in ["audio", "both"]:
        mean_audio = torch.tensor(np.load("static/assets/norm_params/audio_class_mean.npy"), dtype=torch.float32).to(DEVICE)
        std_audio = torch.tensor(np.load("static/assets/norm_params/audio_class_std.npy"), dtype=torch.float32).to(DEVICE)
    print(MODE)

    if MODE == "both":
        text = (text - mean_text) / std_text
        audio = (audio - mean_audio) / std_audio

    elif MODE == "text":
        text = (text - mean_text) / std_text

    elif MODE == "audio":
        audio = (audio - mean_audio) / std_audio

    model.eval()
    with torch.no_grad():
        if mode == "both":
            text, audio = text.to(DEVICE), audio.to(DEVICE)
            outputs = model(text, audio)
        elif mode == "audio":
            audio = audio.to(DEVICE)
            outputs = model(audio)
        elif mode == "text":
            text = text.to(DEVICE)
            outputs = model(text)
        preds = torch.argmax(outputs, dim=1)
    return preds.item()