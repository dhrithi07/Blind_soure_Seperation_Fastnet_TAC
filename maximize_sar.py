import numpy as np
import soundfile as sf
from scipy import signal
from scipy.ndimage import gaussian_filter
import os


def apply_fidelity_injection(mix_file, est_files):
    # --- CALIBRATED FOR 10.36 SIR BASELINE ---
    alpha = 5.5  # Slightly lower than 5.0 to keep SAR higher
    sigma = 0.5  # Smooths the 'edges' of the voices
    floor = 0.012  # THE SECRET: 12% Mix Floor to boost SAR without killing SIR
    # -----------------------------------------

    print(f"--- Injecting Fidelity: Floor {floor} | Alpha {alpha} ---")

    def load_mono(f):
        d, fs = sf.read(f)
        if d.ndim > 1: d = d[:, 0]
        return d, fs

    mix, fs = load_mono(mix_file)
    est1, _ = load_mono(est_files[0])
    est2, _ = load_mono(est_files[1])

    min_len = min(len(mix), len(est1), len(est2))
    mix, est1, est2 = mix[:min_len], est1[:min_len], est2[:min_len]

    # STFT
    nperseg, noverlap = 1024, 768
    f, t, Zxx_mix = signal.stft(mix, fs, nperseg=nperseg, noverlap=noverlap)

    _, _, Zxx_1 = signal.stft(est1, fs, nperseg=nperseg, noverlap=noverlap)
    _, _, Zxx_2 = signal.stft(est2, fs, nperseg=nperseg, noverlap=noverlap)

    # 1. Ratio Mask
    eps = np.finfo(float).eps
    mag_1 = np.abs(Zxx_1) ** alpha
    mag_2 = np.abs(Zxx_2) ** alpha
    mask_1 = mag_1 / (mag_1 + mag_2 + eps)
    mask_2 = mag_2 / (mag_1 + mag_2 + eps)

    # 2. Gaussian Smoothing
    mask_1 = gaussian_filter(mask_1, sigma=sigma)
    mask_2 = gaussian_filter(mask_2, sigma=sigma)

    # 3. APPLY THE FIDELITY FLOOR
    # This fills the "spectral holes" with 12% of the original clean audio
    mask_1 = np.maximum(mask_1, floor)
    mask_2 = np.maximum(mask_2, floor)

    # 4. Apply to Original Mix
    Zxx_out1 = Zxx_mix * mask_1
    Zxx_out2 = Zxx_mix * mask_2

    # 5. Inverse STFT
    _, out1 = signal.istft(Zxx_out1, fs, nperseg=nperseg, noverlap=noverlap)
    _, out2 = signal.istft(Zxx_out2, fs, nperseg=nperseg, noverlap=noverlap)

    def save_norm(fname, audio):
        max_val = np.max(np.abs(audio))
        if max_val > 0: audio = audio / max_val * 0.9
        sf.write(fname, audio, fs)

    save_norm("fasnet_mask_1.wav", out1)
    save_norm("fasnet_mask_2.wav", out2)
    print("✅ Fidelity Injection Complete.")


if __name__ == "__main__":
    mix = "mix_4ch_tac.wav"
    estimates = ["fasnet_mask_1.wav", "fasnet_mask_2.wav"]
    if os.path.exists(mix):
        apply_fidelity_injection(mix, estimates)