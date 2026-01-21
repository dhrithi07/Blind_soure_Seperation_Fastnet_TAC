import matplotlib.pyplot as plt
import soundfile as sf
import numpy as np
import os


def plot_spectrogram(data, fs, ax, title):
    """Helper to plot a log-spectrogram with epsilon safety."""
    # Convert to mono if needed
    if data.ndim > 1: data = data[:, 0]

    # --- FIX FOR "DIVIDE BY ZERO" WARNING ---
    # Add a tiny epsilon so we never take log10(0) in silence
    data = data + 1e-10

    # Plot spectrogram
    Pxx, freqs, bins, im = ax.specgram(data, NFFT=1024, Fs=fs, noverlap=512, cmap='viridis')
    ax.set_title(title)
    ax.set_ylabel("Frequency (Hz)")
    ax.set_xlabel("Time (s)")


def compare_results():
    print("Generating full comparison plots (Reference vs. Raw vs. Masked)...")

    # Define the 3x2 grid layout
    rows = [
        # Row 1: The Goal (Reference/Ground Truth)
        [("Reference Source 1 (Ideal)", "ref1.wav"),
         ("Reference Source 2 (Ideal)", "ref2.wav")],

        # Row 2: The AI's Raw Output
        [("Raw FaSNet 1 (Predicted)", "fasnet_output_1.wav"),
         ("Raw FaSNet 2 (Predicted)", "fasnet_output_2.wav")],

        # Row 3: The Final Polished Result
        [("Masked (Final) 1", "fasnet_masked_1.wav"),
         ("Masked (Final) 2", "fasnet_masked_2.wav")]
    ]

    # Check files
    all_files = [fname for row in rows for _, fname in row]
    if not all(os.path.exists(f) for f in all_files):
        print(f"Error: Missing files. Looked for: {all_files}")
        return

    # Create Figure: 3 rows, 2 columns
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))

    # Store reference lengths to align plots perfectly
    # (Initialize with None placeholders)
    ref_lengths = [None, None]

    # Loop through rows and columns to plot
    for r in range(3):
        for c in range(2):
            title, fname = rows[r][c]
            ax = axes[r, c]

            # Load Data
            data, fs = sf.read(fname)

            # Logic to match lengths
            if r == 0:
                # If this is the Reference row, SAVE the length
                ref_lengths[c] = len(data)
            else:
                # If this is Raw or Masked, CROP to the saved reference length
                # This ensures the time axis (x-axis) looks exactly the same
                if ref_lengths[c] is not None:
                    data = data[:ref_lengths[c]]

            plot_spectrogram(data, fs, ax, title)

    plt.tight_layout()
    print("Plot generated. Opening window...")
    plt.show()


if __name__ == "__main__":
    compare_results()