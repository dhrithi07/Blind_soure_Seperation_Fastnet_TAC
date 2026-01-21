import torch
import functools

# --- CRITICAL FIX FOR PYTORCH 2.6 ---
# We must disable the strict security check for this specific model
# because it was trained in 2020 using older numpy formats.
# This patch forces weights_only=False globally.
print("Applying PyTorch 2.6+ security patch...")
torch.load = functools.partial(torch.load, weights_only=False)

# --- IMPORTS (Must come AFTER the patch) ---
import soundfile as sf
from asteroid.models import BaseModel
import numpy as np
import os
import time


def run_fasnet(input_file):
    print(f"--- Running FaSNet-TAC on {input_file} ---")

    # 1. Load Pre-trained FaSNet-TAC Model
    model_id = "popcornell/FasNetTAC-paper"
    print(f"Loading model '{model_id}'...")

    try:
        # The patch at the top of the file handles the 'weights_only' error automatically now
        model = BaseModel.from_pretrained(model_id)
        model.eval()
    except Exception as e:
        print(f"\nCRITICAL ERROR: Could not download/load model.")
        print(f"Details: {e}")
        return

    # 2. Load Multi-Channel Audio
    # Soundfile reads as (Samples, Channels) -> e.g. (Data, 4)
    data, fs = sf.read(input_file)
    print(f"Loaded audio with shape: {data.shape} and sample rate: {fs}")

    # 3. Prepare Data for Model
    # FaSNet expects shape: (Batch, Microphones, Time)

    # Transpose: (Samples, 4) -> (4, Samples)
    data = data.T

    # Convert to Tensor
    waveform = torch.from_numpy(data).float()

    # SAFETY NORMALIZATION (Input)
    # Scale input to 0.9 to prevent model overload
    max_val = torch.max(torch.abs(waveform))
    if max_val > 0:
        waveform = waveform / max_val * 0.9
        print(f"Input normalized. Peak was: {max_val:.4f}")

    # Add Batch Dimension -> (1, 4, Samples)
    input_tensor = waveform.unsqueeze(0)

    # 4. Run Separation
    print("Separating sources (using spatial cues)...")
    start_time = time.time()

    with torch.no_grad():
        # Model returns: (Batch, Sources, Time)
        estimated_sources = model(input_tensor)

    print(f"--- Processing Time: {time.time() - start_time:.4f} seconds ---")

    # 5. Save Outputs
    # Remove batch dimension -> (Sources, Time)
    output_tensor = estimated_sources.squeeze(0)

    for i in range(output_tensor.shape[0]):
        # Get single source
        out_audio = output_tensor[i].numpy()

        # SAFETY NORMALIZATION (Output)
        # Prevents "Blue Block" clipping
        max_out = np.max(np.abs(out_audio))
        if max_out > 0:
            out_audio = out_audio / max_out * 0.9

        fname = f"fasnet_output_{i + 1}.wav"
        sf.write(fname, out_audio, 16000)
        print(f"Saved: {fname}")


if __name__ == "__main__":
    # Use the relative path since we are running from the project folder
    filename = "mix_4ch_tac.wav"

    # Simple check
    if os.path.exists(filename):
        run_fasnet(filename)
    else:
        print(f"Error: '{filename}' not found.")
        print("Please run your mixing script (Multichannel_Symmetric_Audio_Mixer.py) first.")