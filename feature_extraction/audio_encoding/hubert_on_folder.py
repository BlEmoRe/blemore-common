import os
from pathlib import Path
import torch
import torchaudio
import numpy as np
from s3prl.hub import hubert_base, hubert_large_ll60k

# === CONFIGURATION ===
USE_LARGE_MODEL = True  # Set to True for hubert_large
INPUT_DIR = Path("/media/user/Seagate Hub/mixed_emotion_challenge/wav_files")
OUTPUT_DIR = Path("/media/user/Seagate Hub/mixed_emotion_challenge/audio_encodings/hubert_large")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# === LOAD MODEL ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Loading model on {device}...")

model = hubert_large_ll60k() if USE_LARGE_MODEL else hubert_base()
model = model.to(device)
model.eval()

# === PROCESS EACH .wav FILE ===
for wav_path in INPUT_DIR.glob("*.wav"):
    print(f"Processing {wav_path.name}...")
    try:
        # Load and resample
        wav, sr = torchaudio.load(wav_path)
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
        wav = wav.to(device)

        # Extract features
        with torch.inference_mode():
            features = model(wav)

        # Get last hidden state, mean pool across time
        last_hidden = features["last_hidden_state"].squeeze(0)  # shape: [T, 768]
        # pooled = last_hidden.mean(dim=0).cpu().numpy()          # shape: [768]

        # Save
        out_path = OUTPUT_DIR / (wav_path.stem + ".npy")
        np.save(out_path, last_hidden.cpu())
        print(f"✅ Processed {wav_path.name}")

    except Exception as e:
        print(f"❌ Failed {wav_path.name}: {e}")
