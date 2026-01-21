import numpy as np
import soundfile as sf
from scipy import signal
import os


def apply_aggressive_masking(mix_file, est_files, alpha=0.5):
    print(f"--- Applying Aggressive Masking (Alpha={alpha}) ---")

    # 1. Load Files
    mix, fs = sf.read(mix_file)
    est1, _ = sf.read(est_files[0])
    est2, _ = sf.read(est_files[1])

    # Use Channel 0 of the mix if multi-channel
    if mix.ndim > 1: mix = mix[:, 0]

    # Crop to shortest length
    min_len = min(len(mix), len(est1), len(est2))
    mix = mix[:min_len]
    est1 = est1[:min_len]
    est2 = est2[:min_len]

    # 2. Convert to Spectrograms (STFT)
    # nperseg=1024 / noverlap=768 gives higher time resolution for speech
    f, t, Zxx_mix = signal.stft(mix, fs, nperseg=1024, noverlap=768)
    _, _, Zxx_1 = signal.stft(est1, fs, nperseg=1024, noverlap=768)
    _, _, Zxx_2 = signal.stft(est2, fs, nperseg=1024, noverlap=768)

    # 3. Create the "Aggressive Masks"
    eps = np.finfo(float).eps

    # Calculate Magnitude and apply Power (Alpha)
    # Alpha > 1.0 pushes quiet parts to zero and loud parts to 1.
    mag_1 = np.abs(Zxx_1) ** alpha
    mag_2 = np.abs(Zxx_2) ** alpha

    # The mask calculation
    mask_1 = mag_1 / (mag_1 + mag_2 + eps)
    mask_2 = mag_2 / (mag_1 + mag_2 + eps)

    # 4. Apply Mask to the ORIGINAL Mix
    Zxx_out1 = Zxx_mix * mask_1
    Zxx_out2 = Zxx_mix * mask_2

    # 5. Convert back to Audio (ISTFT)
    _, out1 = signal.istft(Zxx_out1, fs, nperseg=1024, noverlap=768)
    _, out2 = signal.istft(Zxx_out2, fs, nperseg=1024, noverlap=768)

    # 6. Normalize and Save
    def save_norm(fname, audio):
        max_val = np.max(np.abs(audio))
        if max_val > 0: audio = audio / max_val * 0.9
        sf.write(fname, audio, fs)
        print(f"Saved: {fname}")

    save_norm("fasnet_masked_1.wav", out1)
    save_norm("fasnet_masked_2.wav", out2)


if __name__ == "__main__":
    mix = "mix_4ch_tac.wav"
    estimates = ["fasnet_output_1.wav", "fasnet_output_2.wav"]

    if os.path.exists(mix):
        # TRY THIS: Alpha 2.0 is standard.
        # If SIR is still low, try Alpha=4.0 or 10.0
        apply_aggressive_masking(mix, estimates, alpha=3.0)
    else:
        print("Error: Input files not found.")