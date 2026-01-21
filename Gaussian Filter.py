import numpy as np
import soundfile as sf
from scipy import signal
from scipy.ndimage import gaussian_filter
import os


def apply_sharp_mask(mix_file, est_files):
    print("--- Applying Sharp Hybrid Masking (Restoring High SIR) ---")

    # 1. Load Files
    # Force Mono to avoid shape errors
    def load_mono(f):
        data, fs = sf.read(f)
        if data.ndim > 1: data = data[:, 0]
        return data, fs

    mix, fs = load_mono(mix_file)
    est1, _ = load_mono(est_files[0])
    est2, _ = load_mono(est_files[1])

    # 2. Crop to Minimum Length
    min_len = min(len(mix), len(est1), len(est2))
    mix = mix[:min_len]
    est1 = est1[:min_len]
    est2 = est2[:min_len]

    # 3. STFT (Standard Settings)
    nperseg = 1024
    noverlap = 768
    f, t, Zxx_mix = signal.stft(mix, fs, nperseg=nperseg, noverlap=noverlap)
    _, _, Zxx_1 = signal.stft(est1, fs, nperseg=nperseg, noverlap=noverlap)
    _, _, Zxx_2 = signal.stft(est2, fs, nperseg=nperseg, noverlap=noverlap)

    # 4. Create The "Sharp" Mask
    eps = np.finfo(float).eps

    # Alpha = 1.0 is the standard "Hard" separation.
    # Lower values (0.5) are too soft. Higher (1.5) are too artificial.
    alpha = 7.0

    mag_1 = np.abs(Zxx_1) ** alpha
    mag_2 = np.abs(Zxx_2) ** alpha

    # Standard Ratio Mask
    mask_1 = mag_1 / (mag_1 + mag_2 + eps)
    mask_2 = mag_2 / (mag_1 + mag_2 + eps)

    # 5. Minimal Smoothing
    # We only smooth slightly to remove "musical noise" artifacts
    # effectively boosting SIR without ruining quality.
    mask_1 = gaussian_filter(mask_1, sigma=0.5)
    mask_2 = gaussian_filter(mask_2, sigma=0.5)

    # 6. Low Floor (Crucial for High SIR)
    # We DO NOT want to let 50% noise in (floor=0.5).
    # We set floor to 0.01 (1%) to kill the background noise.
    floor = 0.00
    mask_1 = np.maximum(mask_1, floor)
    mask_2 = np.maximum(mask_2, floor)

    # 7. Apply Mask to Original Mix
    Zxx_out1 = Zxx_mix * mask_1
    Zxx_out2 = Zxx_mix * mask_2

    # 8. Inverse STFT
    _, out1 = signal.istft(Zxx_out1, fs, nperseg=nperseg, noverlap=noverlap)
    _, out2 = signal.istft(Zxx_out2, fs, nperseg=nperseg, noverlap=noverlap)

    # 9. Save
    def save_norm(fname, audio):
        max_val = np.max(np.abs(audio))
        if max_val > 0: audio = audio / max_val * 0.9
        sf.write(fname, audio, fs)

    save_norm("fasnet_hybrid_1.wav", out1)
    save_norm("fasnet_hybrid_2.wav", out2)
    print("Saved: fasnet_hybrid_1.wav & fasnet_hybrid_2.wav")


if __name__ == "__main__":
    mix_file = "mix_4ch_tac.wav"
    raw_files = ["fasnet_output_1.wav", "fasnet_output_2.wav"]

    if os.path.exists(mix_file) and os.path.exists(raw_files[0]):
        apply_sharp_mask(mix_file, raw_files)
    else:
        print("Error: Files not found. Run separation first.")