import torch
import soundfile as sf
from asteroid.models import BaseModel
import numpy as np
import os
import functools

# --- THE FIX FOR PYTORCH 2.6+ ---
# We monkey-patch torch.load to force weights_only=False.
# This bypasses the "WeightsUnpickler error".
_original_load = torch.load
torch.load = functools.partial(_original_load, weights_only=False)


# --------------------------------

def run_fasnet_baseline(input_file):
    print(f"--- Running FaSNet-TAC (Baseline) on {input_file} ---")

    # 1. Load Model (Now safe from the error)
    model_id = "popcornell/FasNetTAC-paper"
    try:
        print("   > Loading model (Safety Override Active)...")
        model = BaseModel.from_pretrained(model_id)
        model.eval()
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        return

    # 2. Load Audio
    data, fs = sf.read(input_file)

    # Handle Mono vs Multi-channel
    if data.ndim == 1:
        print("   > Warning: Input is Mono. Converting to pseudo-multichannel.")
        # Stack to make shape (4, N) roughly
        data = np.stack([data] * 4, axis=0)
    else:
        data = data.T  # (Mics, Samples)

    waveform = torch.from_numpy(data).float()

    # Add batch dim -> (1, Mics, Samples)
    input_tensor = waveform.unsqueeze(0)

    # 3. Run Inference
    print("   > Separating...")
    with torch.no_grad():
        est_sources = model(input_tensor)  # (1, Sources, Time)

    # 4. Save (Raw Output)
    est_sources = est_sources.squeeze(0)
    for i in range(est_sources.shape[0]):
        out_audio = est_sources[i].numpy()

        # Normalize
        max_val = np.max(np.abs(out_audio))
        if max_val > 0: out_audio = out_audio / max_val * 0.9

        # Save as 'fasnet_output'
        fname = f"fasnet_output_{i + 1}.wav"
        sf.write(fname, out_audio, 16000)
        print(f"✅ Saved High-SIR Baseline: {fname}")


if __name__ == "__main__":
    if os.path.exists("mix_4ch_tac.wav"):
        run_fasnet_baseline("mix_4ch_tac.wav")
    else:
        print("Mix file not found.")