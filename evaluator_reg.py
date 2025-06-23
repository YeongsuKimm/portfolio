import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class StreamerDataset(Dataset):
    def __init__(self, root_dir, mode="both", norm_dir=None):
        self.root_dir = root_dir
        self.streamers = os.listdir(root_dir)
        self.mode = mode
        self.data = []
        
        self.norm_params = {}
        if norm_dir:
            if mode in ["both", "text"]:
                self.norm_params['text_mean'] = torch.tensor(np.load(f"{norm_dir}/text_reg_mean.npy"), dtype=torch.float32)
                self.norm_params['text_std'] = torch.tensor(np.load(f"{norm_dir}/text_reg_std.npy"), dtype=torch.float32)
            if mode in ["both", "audio"]:
                self.norm_params['audio_mean'] = torch.tensor(np.load(f"{norm_dir}/audio_reg_mean.npy"), dtype=torch.float32)
                self.norm_params['audio_std'] = torch.tensor(np.load(f"{norm_dir}/audio_reg_std.npy"), dtype=torch.float32)
            if os.path.exists(f"{norm_dir}/label_mean.npy") and os.path.exists(f"{norm_dir}/label_std.npy"):
                self.norm_params['label_mean'] = torch.tensor(np.load(f"{norm_dir}/label_mean.npy"), dtype=torch.float32)
                self.norm_params['label_std'] = torch.tensor(np.load(f"{norm_dir}/label_std.npy"), dtype=torch.float32)

        for streamer in self.streamers:
            streamer_path = os.path.join(root_dir, streamer)
            metadata_path = os.path.join(streamer_path, "metadata.h5")
            if not os.path.exists(metadata_path):
                continue
            with h5py.File(metadata_path, "r") as f:
                if "tensor" in f:
                    target = f["tensor"][2]
                else:
                    continue

            text_files = sorted([f for f in os.listdir(streamer_path) if f.startswith("text_") and f.endswith(".h5")])
            audio_files = sorted([f for f in os.listdir(streamer_path) if f.startswith("audio_") and f.endswith(".h5")])

            if mode == "both":
                for text_file, audio_file in zip(text_files, audio_files):
                    self.data.append((os.path.join(streamer_path, text_file), os.path.join(streamer_path, audio_file), target))
            elif mode == "text":
                for text_file in text_files:
                    self.data.append((os.path.join(streamer_path, text_file), target))
            elif mode == "audio":
                for audio_file in audio_files:
                    self.data.append((os.path.join(streamer_path, audio_file), target))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if self.mode == "both":
            text_path, audio_path, target = self.data[idx]
            text_feat = self.load_h5_features(text_path)
            audio_feat = self.load_h5_features(audio_path)

            # Normalize features if stats are loaded
            if 'text_mean' in self.norm_params and 'text_std' in self.norm_params:
                text_feat = (text_feat - self.norm_params['text_mean']) / self.norm_params['text_std']
            if 'audio_mean' in self.norm_params and 'audio_std' in self.norm_params:
                audio_feat = (audio_feat - self.norm_params['audio_mean']) / self.norm_params['audio_std']
            target = torch.tensor([target], dtype=torch.float32)
            if 'label_mean' in self.norm_params and 'label_std' in self.norm_params:
                target = (target - self.norm_params['label_mean']) / self.norm_params['label_std']
            return text_feat, audio_feat, target

        elif self.mode == "text":
            text_path, target = self.data[idx]
            text_feat = self.load_h5_features(text_path)
            if 'text_mean' in self.norm_params and 'text_std' in self.norm_params:
                text_feat = (text_feat - self.norm_params['text_mean']) / self.norm_params['text_std']
            target = torch.tensor([target], dtype=torch.float32)
            if 'label_mean' in self.norm_params and 'label_std' in self.norm_params:
                target = (target - self.norm_params['label_mean']) / self.norm_params['label_std']
            return text_feat, target

        elif self.mode == "audio":
            audio_path, target = self.data[idx]
            audio_feat = self.load_h5_features(audio_path)
            if 'audio_mean' in self.norm_params and 'audio_std' in self.norm_params:
                audio_feat = (audio_feat - self.norm_params['audio_mean']) / self.norm_params['audio_std']
            target = torch.tensor([target], dtype=torch.float32)
            if 'label_mean' in self.norm_params and 'label_std' in self.norm_params:
                target = (target - self.norm_params['label_mean']) / self.norm_params['label_std']
            return audio_feat, target

    def load_h5_features(self, file_path):
        with h5py.File(file_path, "r") as f:
            if "tensor" in f:
                data = f["tensor"][()]
        return torch.tensor(data, dtype=torch.float32).squeeze(0)
    
class TextOnlyRegressor(nn.Module):
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

class AudioOnlyRegressor(nn.Module):
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

class MultiModalRegressor(nn.Module):
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

def evaluate_model(model, dataloader, mode="both", norm_dir=None, device="cuda"):
    model.eval()
    model.to(device)
    criterion = torch.nn.MSELoss()

    all_preds = []
    all_targets = []
    total_loss = 0.0

    with torch.no_grad():
        for batch in dataloader:
            if mode == "both":
                text, audio, target = batch
                text, audio, target = text.to(device), audio.to(device), target.to(device)
                outputs = model(text, audio).squeeze()
            else:
                inputs, target = batch
                inputs, target = inputs.to(device), target.to(device)
                outputs = model(inputs).squeeze()

            if norm_dir:
                label_mean = torch.tensor(np.load(f"{norm_dir}/label_mean.npy")).to(device)
                label_std = torch.tensor(np.load(f"{norm_dir}/label_std.npy")).to(device)
                outputs = outputs * label_std + label_mean
                target = target * label_std + label_mean

            loss = criterion(outputs, target.squeeze())
            total_loss += loss.item()

            all_preds.append(outputs.cpu())
            all_targets.append(target.cpu())

    avg_loss = total_loss / len(dataloader)
    print(f"Validation MSE Loss: {avg_loss:.4f}")
    return torch.cat(all_preds), torch.cat(all_targets)

def load_model(mode, device):
    TEXT_DIM = AUDIO_DIM = 768
    HIDDEN_DIM = 128
    OUTPUT_DIM = 1

    if mode == "text":
        model = TextOnlyRegressor(TEXT_DIM, HIDDEN_DIM, OUTPUT_DIM)
        model_path = "static/assets/models/text_only_reg_model_normalized_t70-3.pth"
    elif mode == "audio":
        model = AudioOnlyRegressor(AUDIO_DIM, HIDDEN_DIM, OUTPUT_DIM)
        model_path = "static/assets/models/audio_only_reg_model_normalized_t70-3.pth"
    else:
        model = MultiModalRegressor(TEXT_DIM, AUDIO_DIM, HIDDEN_DIM, OUTPUT_DIM)
        model_path = "static/assets/models/streamer_reg_model_normalized_t70-3.pth"

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def get_preds_reg(mode, text="False", audio="False"):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MODE = mode 
    model = load_model(mode=MODE, device=DEVICE)
    print(MODE)
    if MODE in ["text", "both"]:
        mean_text = torch.tensor(np.load("static/assets/norm_params/text_reg_mean.npy"), dtype=torch.float32).to(DEVICE)
        std_text = torch.tensor(np.load("static/assets/norm_params/text_reg_std.npy"), dtype=torch.float32).to(DEVICE)
    if MODE in ["audio", "both"]:
        mean_audio = torch.tensor(np.load("static/assets/norm_params/audio_reg_mean.npy"), dtype=torch.float32).to(DEVICE)
        std_audio = torch.tensor(np.load("static/assets/norm_params/audio_reg_std.npy"), dtype=torch.float32).to(DEVICE)

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
            outputs = model(text, audio).squeeze()
        elif mode == "audio":
            audio = audio.to(DEVICE)
            outputs = model(audio)
        elif mode == "text":
            text = text.to(DEVICE)
            outputs = model(text)
        label_mean = torch.tensor(np.load("static/assets/norm_params/label_mean.npy")).to(DEVICE)
        label_std = torch.tensor(np.load("static/assets/norm_params/label_std.npy")).to(DEVICE)
        outputs = outputs * label_std + label_mean
        
    return round(outputs.item(),4)