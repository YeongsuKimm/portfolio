import os
import json
import torch
import tqdm
import torchaudio
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaModel, AutoProcessor, AutoModel
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchaudio.transforms as transforms
import whisper

# Device setup
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load frozen feature extractors ONCE
roberta_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
roberta_model = RobertaModel.from_pretrained("roberta-base",output_hidden_states = True).to(DEVICE).eval()

# roberta_model.config.output_hidden_states = True;

wav2vec_processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base")
wav2vec_model = AutoModel.from_pretrained("facebook/wav2vec2-base", output_hidden_states = True).to(DEVICE).eval()

def extract_text_features(text):
    tokens = roberta_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(DEVICE)
    with torch.no_grad():
        output = roberta_model(**tokens)
    return output.hidden_states[-2][:,0,:]  # Shape: [1, 768]

def extract_audio_features(audio_path):
    waveform, sample_rate = torchaudio.load(audio_path)

    # Convert stereo to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample if necessary
    target_sample_rate = 16000
    if sample_rate != target_sample_rate:
        waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)(waveform)

    # Process input for Wav2Vec2
    inputs = wav2vec_processor(waveform.squeeze(0), sampling_rate=target_sample_rate, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        output = wav2vec_model(**inputs)
    return output.hidden_states[-2][:,0,:]  # Shape: [1, 768]


def transcribe_audio(audio_path):
    """Transcribes the given audio file using Whisper and saves the text."""
    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    text = result["text"]
    return text
