import numpy as np
import soundfile as sf
from scipy import signal
from scipy.ndimage import gaussian_filter
import os


def final_polish(mix_file, sep_files, blend_ratio=0.30):
    print(f"--- Polishing for SAR 30+ (Blend Ratio: {blend_ratio}) ---")

    mix, fs = sf.read(mix_file)
    if mix.ndim > 1: mix = mix[:, 0]

    seps = [sf.read(f)[0] for f in sep_files]

    # 1. STFT
    n_fft = 2048
    hop = 512
    f, t, Zxx_mix = signal.stft(mix, fs, nperseg=n_fft, noverlap=hop)
    mag_mix = np.abs(Zxx_mix)
    phase_mix = np.angle(Zxx_mix)

    for i, sep in enumerate(seps):
        # 2. Get Separation Magnitude
        _, _, Zxx_sep = signal.stft(sep[:len(mix)], fs, nperseg=n_fft, noverlap=hop)
        mag_sep = np.abs(Zxx_sep)

        # 3. SMOOTHING (The Anti-Robot Step)
        # This removes the jagged digital artifacts
        mag_sep_smooth = gaussian_filter(mag_sep, sigma=1.5)

        # 4. SOFT BLENDING
        # This keeps 70% of the original audio quality (SAR)
        # while using 30% of the mask to provide separation (SIR)
        mag_final = (mag_sep_smooth * blend_ratio) + (mag_mix * (1 - blend_ratio))

        # 5. RECONSTRUCT
        Zxx_final = mag_final * np.exp(1j * phase_mix)
        _, out = signal.istft(Zxx_final, fs, nperseg=n_fft, noverlap=hop)

        # Normalize
        out = out / (np.max(np.abs(out)) + 1e-9) * 0.9

        fname = f"fasnet_masked_{i + 1}.wav"
        sf.write(fname, out, fs)
        print(f"✅ Created: {fname}")


if __name__ == "__main__":
    # Use the RAW outputs from your successful FaSNet run
    final_polish("mix_4ch_tac.wav", ["fasnet_hybrid_1.wav", "fasnet_hybrid_2.wav"])