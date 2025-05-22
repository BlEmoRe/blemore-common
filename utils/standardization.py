import numpy as np
import os

def compute_train_stats(filenames, encoding_dir):
    total_sum = None
    total_sq_sum = None
    total_count = 0

    for fname in filenames:
        x = np.load(os.path.join(encoding_dir, f"{fname}.npy"))  # shape (T, D)
        if total_sum is None:
            D = x.shape[1]
            total_sum = np.zeros(D)
            total_sq_sum = np.zeros(D)

        total_sum += x.sum(axis=0)           # sum over time
        total_sq_sum += (x ** 2).sum(axis=0) # sum of squares
        total_count += x.shape[0]            # total number of frames (T)

    mean = total_sum / total_count
    var = total_sq_sum / total_count - mean ** 2
    std = np.sqrt(np.maximum(var, 1e-8))  # prevent divide-by-zero

    return mean, std


def main():
    enc_path = "/home/tim/Work/quantum/data/blemore/encoded_videos/CLIP_npy/"
    train_metadata = "/home/tim/Work/quantum/data/blemore/train_metadata.csv"



