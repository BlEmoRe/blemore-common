from s3prl.hub import hubert_base
import torchaudio

# Load model
model = hubert_base()

# Load a WAV file
waveform, sample_rate = torchaudio.load("/media/user/Seagate Hub/mixed_emotion_challenge/wav_files/A303_ang_int1_ver1.wav")
if sample_rate != 16000:
    waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)

# Shape: (batch=1, channel=1, time)
features = model(waveform)  # Output: a list of hidden states
last_hidden = features["last_hidden_state"].squeeze(0)    # (time, feature_dim)

print("Shape of extracted features:", last_hidden.shape)
