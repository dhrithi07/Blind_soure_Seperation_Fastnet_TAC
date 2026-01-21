import matplotlib.pyplot as plt
import soundfile as sf
import numpy as np
import os


def plot_spectrogram(data, fs, ax, title):
    """Helper to plot a log-spectrogram"""
    # Convert to mono if needed
    if data.ndim > 1: data = data[:, 0]

    Pxx, freqs, bins, im = ax.specgram(data, NFFT=1024, Fs=fs, noverlap=512)
    ax.set_title(title)
    ax.set_ylabel("Frequency (Hz)")
    ax.set_xlabel("Time (s)")


def visualize_results():
    files = {
        "Input Mix (Mic 1)": "mix_4ch_tac.wav",
        "Target Source 1 (Ref)": "ref1.wav",
        "FaSNet Output 1": "fasnet_output_1.wav",
        "Target Source 2 (Ref)": "ref2.wav",
        "FaSNet Output 2": "fasnet_output_2.wav"
    }

    # Check files
    if not all(os.path.exists(f) for f in files.values()):
        print("Missing files. Run separation first.")
        return

    plt.figure(figsize=(15, 10))

    # We will create a 3x2 grid (Input at top, then Source 1 col, Source 2 col)
    # Row 1: Input Mix (Spanning both columns)
    data_mix, fs = sf.read(files["Input Mix (Mic 1)"])
    if data_mix.ndim > 1: data_mix = data_mix[:, 0]  # Take first mic

    ax1 = plt.subplot(3, 1, 1)
    plot_spectrogram(data_mix, fs, ax1, "Input Mixture (What the AI heard)")

    # Row 2: Source 1 Comparison
    data_ref1, _ = sf.read(files["Target Source 1 (Ref)"])
    data_est1, _ = sf.read(files["FaSNet Output 1"])

    ax2 = plt.subplot(3, 2, 3)
    plot_spectrogram(data_ref1, fs, ax2, "Reference Source 1 (Ideal)")

    ax3 = plt.subplot(3, 2, 4)
    plot_spectrogram(data_est1, fs, ax3, "FaSNet Output 1 (Predicted)")

    # Row 3: Source 2 Comparison
    data_ref2, _ = sf.read(files["Target Source 2 (Ref)"])
    data_est2, _ = sf.read(files["FaSNet Output 2"])

    ax4 = plt.subplot(3, 2, 5)
    plot_spectrogram(data_ref2, fs, ax4, "Reference Source 2 (Ideal)")

    ax5 = plt.subplot(3, 2, 6)
    plot_spectrogram(data_est2, fs, ax5, "FaSNet Output 2 (Predicted)")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    visualize_results()