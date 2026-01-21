import numpy as np
import soundfile as sf
from scipy import signal
from scipy.ndimage import gaussian_filter
import os


def apply_sar_polish(mix_file, mvdr_files):
    # --- CALIBRATED FOR SAR 33 / SIR 9 ---
    alpha = 1.3  # Slightly increased to keep separation sharp
    sigma = 1.0  # Smoothing to remove digital 'jitter'
    floor = 0.40  # Increased to 40% to aggressively target SAR 33
    # -------------------------------------

    print(f"--- Polishing MVDR for SAR 30+ | Floor: {floor} ---")

    def load_mono(f):
        d, fs = sf.read(f)
        if d.ndim > 1: d = d[:, 0]
        return d, fs

    mix, fs = load_mono(mix_file)
    m1, _ = load_mono(mvdr_files[0])
    m2, _ = load_mono(mvdr_files[1])

    min_len = min(len(mix), len(m1), len(m2))
    mix, m1, m2 = mix[:min_len], m1[:min_len], m2[:min_len]

    # STFT
    nperseg, noverlap = 1024, 768
    f, t, Zxx_mix = signal.stft(mix, fs, nperseg=nperseg, noverlap=noverlap)
    _, _, Zxx_m1 = signal.stft(m1, fs, nperseg=nperseg, noverlap=noverlap)
    _, _, Zxx_m2 = signal.stft(m2, fs, nperseg=nperseg, noverlap=noverlap)

    # 1. Soft Ratio Mask
    eps = np.finfo(float).eps
    mag_1 = np.abs(Zxx_m1) ** alpha
    mag_2 = np.abs(Zxx_m2) ** alpha

    mask_1 = mag_1 / (mag_1 + mag_2 + eps)
    mask_2 = mag_2 / (mag_1 + mag_2 + eps)

    # 2. Gaussian Smoothing (Fixing the previous error)
    mask_1 = gaussian_filter(mask_1, sigma=sigma)
    mask_2 = gaussian_filter(mask_2, sigma=sigma)

    # 3. Apply Quality Floor
    mask_1 = np.maximum(mask_1, floor)
    mask_2 = np.maximum(mask_2, floor)

    # 4. Apply to MVDR signals
    # This keeps the spatial benefits of MVDR but softens the artifacts
    Zxx_out1 = Zxx_m1 * mask_1
    Zxx_out2 = Zxx_m2 * mask_2

    # 5. Inverse STFT
    _, out1 = signal.istft(Zxx_out1, fs, nperseg=nperseg, noverlap=noverlap)
    _, out2 = signal.istft(Zxx_out2, fs, nperseg=nperseg, noverlap=noverlap)

    def save_norm(fname, audio):
        max_val = np.max(np.abs(audio))
        if max_val > 0: audio = audio / max_val * 0.9
        sf.write(fname, audio, fs)

    save_norm("fasnet_final_1.wav", out1)
    save_norm("fasnet_final_2.wav", out2)
    print("✅ Polish Complete.")


if __name__ == "__main__":
    mvdr_files = ["fasnet_hybrid_1.wav", "fasnet_hybrid_2.wav"]
    if os.path.exists(mvdr_files[0]):
        apply_sar_polish("mix_4ch_tac.wav", mvdr_files)
    else:
        print("Error: MVDR files (fasnet_hybrid) not found. Run MVDR script first.")