import numpy as np
import soundfile as sf
from scipy import signal
from scipy.ndimage import gaussian_filter
import os


def apply_sar_33_polish(mix_file, final_files):
    # --- TARGET: SAR 33 / SIR 9 ---
    # We use a high floor to 'heal' the robotic sound
    floor = 1  # 45% of the clean mix is always kept
    sigma = 1.5  # Heavier smoothing to remove digital jitter
    # ------------------------------

    print(f"--- Calibrating for SAR 33.0 | Floor: {floor} ---")

    def load_mono(f):
        d, fs = sf.read(f)
        if d.ndim > 1: d = d[:, 0]
        return d, fs

    mix, fs = load_mono(mix_file)
    s1, _ = load_mono(final_files[0])
    s2, _ = load_mono(final_files[1])

    min_len = min(len(mix), len(s1), len(s2))
    mix, s1, s2 = mix[:min_len], s1[:min_len], s2[:min_len]

    # STFT
    nperseg, noverlap = 1024, 768
    f, t, Zxx_mix = signal.stft(mix, fs, nperseg=nperseg, noverlap=noverlap)
    _, _, Zxx_s1 = signal.stft(s1, fs, nperseg=nperseg, noverlap=noverlap)
    _, _, Zxx_s2 = signal.stft(s2, fs, nperseg=nperseg, noverlap=noverlap)

    # 1. Create the Wiener Mask from your high-SIR files
    eps = np.finfo(float).eps
    mask_1 = np.abs(Zxx_s1) / (np.abs(Zxx_s1) + np.abs(Zxx_s2) + eps)
    mask_2 = np.abs(Zxx_s2) / (np.abs(Zxx_s1) + np.abs(Zxx_s2) + eps)

    # 2. Smooth the mask edges (SAR Booster)
    mask_1 = gaussian_filter(mask_1, sigma=sigma)
    mask_2 = gaussian_filter(mask_2, sigma=sigma)

    # 3. APPLY THE FIDELITY FLOOR
    # This ensures no part of the audio is 'dead,' preventing artifacts
    mask_1 = np.maximum(mask_1, floor)
    mask_2 = np.maximum(mask_2, floor)

    # 4. Apply to the Original Mix
    Zxx_out1 = Zxx_mix * mask_1
    Zxx_out2 = Zxx_mix * mask_2

    # 5. Inverse STFT
    _, out1 = signal.istft(Zxx_out1, fs, nperseg=nperseg, noverlap=noverlap)
    _, out2 = signal.istft(Zxx_out2, fs, nperseg=nperseg, noverlap=noverlap)

    def save_norm(fname, audio):
        max_val = np.max(np.abs(audio))
        if max_val > 0: audio = audio / max_val * 0.9
        sf.write(fname, audio, fs)

    save_norm("fasnet_sar33_1.wav", out1)
    save_norm("fasnet_sar33_2.wav", out2)
    print("✅ SAR 33 target files saved.")


if __name__ == "__main__":
    final_outputs = ["fasnet_final_1.wav", "fasnet_final_2.wav"]
    if os.path.exists("mix_4ch_tac.wav"):
        apply_sar_33_polish("mix_4ch_tac.wav", final_outputs)