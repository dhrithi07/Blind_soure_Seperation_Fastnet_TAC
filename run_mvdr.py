import numpy as np
import soundfile as sf
from scipy import signal
import os


def apply_mvdr_hybrid(mix_file, est_files):
    print("--- Applying MVDR-Hybrid (Spatial + Spectral) ---")

    mix, fs = sf.read(mix_file)
    if mix.ndim == 1:
        print("Error: Mix must be 4-channel.")
        return

    est1, _ = sf.read(est_files[0])
    est2, _ = sf.read(est_files[1])

    min_len = min(len(mix), len(est1), len(est2))
    mix = mix[:min_len].T
    ests = np.stack([est1[:min_len], est2[:min_len]], axis=0)

    n_fft, hop = 1024, 512
    f, t, Zxx_mix = signal.stft(mix, fs, nperseg=n_fft, noverlap=hop)
    num_mics, num_freq, num_frames = Zxx_mix.shape

    # Pre-calculate Spectrograms for Masking
    _, _, Zxx_s1 = signal.stft(ests[0], fs, nperseg=n_fft, noverlap=hop)
    _, _, Zxx_s2 = signal.stft(ests[1], fs, nperseg=n_fft, noverlap=hop)

    # THE KEY: Create a Wiener Mask to help the MVDR
    mag1, mag2 = np.abs(Zxx_s1) ** 2, np.abs(Zxx_s2) ** 2
    mask1 = mag1 / (mag1 + mag2 + 1e-9)
    mask2 = mag2 / (mag1 + mag2 + 1e-9)
    masks = [mask1, mask2]

    outputs = []
    for s_idx in range(2):
        print(f"   > Processing Source {s_idx + 1}...")
        target_mask = masks[s_idx]
        processed_spec = np.zeros((num_freq, num_frames), dtype=complex)

        for f_idx in range(num_freq):
            X = Zxx_mix[:, f_idx, :]  # (4, Frames)

            # Use the Mask to estimate the Noise Covariance (Phi_NN)
            # This tells the MVDR exactly what to 'cancel'
            noise_mask = 1 - target_mask[f_idx, :]
            X_noise = X * noise_mask
            Rnn = np.dot(X_noise, X_noise.conj().T) / (np.sum(noise_mask) + 1e-9)
            Rnn += np.eye(num_mics) * 1e-5  # Regularization

            # Estimate Steering Vector from the Target-Masked signal
            X_target = X * target_mask[f_idx, :]
            Rss = np.dot(X_target, X_target.conj().T) / (np.sum(target_mask[f_idx, :]) + 1e-9)
            eigvals, eigvecs = np.linalg.eigh(Rss)
            d = eigvecs[:, -1]  # Principal component is the steering vector

            try:
                Rinv = np.linalg.inv(Rnn)
                w = np.dot(Rinv, d) / (np.dot(d.conj().T, np.dot(Rinv, d)) + 1e-9)

                # Apply MVDR beamformer
                spatial_out = np.dot(w.conj().T, X)

                # POST-FILTERING: Apply a light mask to ensure they don't sound the same
                processed_spec[f_idx, :] = spatial_out * (target_mask[f_idx, :] * 0.8 + 0.2)

            except np.linalg.LinAlgError:
                processed_spec[f_idx, :] = Zxx_s1[f_idx, :] if s_idx == 0 else Zxx_s2[f_idx, :]

        outputs.append(processed_spec)

    for i, spec in enumerate(outputs):
        _, audio = signal.istft(spec, fs, nperseg=n_fft, noverlap=hop)
        audio = audio / (np.max(np.abs(audio)) + 1e-9) * 0.9
        sf.write(f"fasnet_hybrid_{i + 1}.wav", audio, fs)


if __name__ == "__main__":
    apply_mvdr_hybrid("mix_4ch_tac.wav", ["fasnet_output_1.wav", "fasnet_output_2.wav"])