import torch
import functools

# --- PYTORCH 2.6 SECURITY PATCH ---
print("Applying PyTorch 2.6+ security patch...")
torch.load = functools.partial(torch.load, weights_only=False)

import soundfile as sf
from asteroid.models import BaseModel
import numpy as np
import os
import time


def run_shift_voting(input_file, shifts=[0, 50, 100, 150]):
    """
    Runs FaSNet multiple times with small time-shifts to smooth out artifacts.
    shifts: List of sample offsets to shift the input by.
    """
    print(f"--- Running FaSNet with Shift-Voting (Shifts: {shifts}) ---")

    # 1. Load Model
    model_id = "popcornell/FasNetTAC-paper"
    try:
        model = BaseModel.from_pretrained(model_id)
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 2. Load Audio
    data, fs = sf.read(input_file)
    data = data.T  # (Mics, Samples)
    waveform = torch.from_numpy(data).float()

    # Normalize Input
    max_val = torch.max(torch.abs(waveform))
    if max_val > 0: waveform = waveform / max_val * 0.9

    # 3. The Shift-Voting Loop
    accumulated_output = None
    count = 0

    with torch.no_grad():
        for shift in shifts:
            print(f"   > Processing shift: {shift} samples...")

            # A. Shift Input (Pad start with zeros)
            # Shape: (Mics, Samples + shift)
            padded_input = torch.nn.functional.pad(waveform, (shift, 0))
            input_tensor = padded_input.unsqueeze(0)  # Add batch dim

            # B. Run Inference
            est_source = model(input_tensor)  # (1, Sources, Time)

            # C. Un-Shift Output (Crop the start)
            # We cut off the first 'shift' samples to realign
            est_aligned = est_source[..., shift:]

            # Ensure lengths match exactly (model might have padded end)
            target_len = waveform.shape[-1]
            est_aligned = est_aligned[..., :target_len]

            # D. Accumulate
            if accumulated_output is None:
                accumulated_output = est_aligned
            else:
                accumulated_output += est_aligned
            count += 1

    # 4. Average
    print("   > Averaging results...")
    final_output = accumulated_output / count
    final_output = final_output.squeeze(0)

    # 5. Save
    for i in range(final_output.shape[0]):
        out_audio = final_output[i].numpy()

        # Normalize Output
        max_out = np.max(np.abs(out_audio))
        if max_out > 0: out_audio = out_audio / max_out * 0.9

        fname = f"fasnet_shift_output_{i + 1}.wav"
        sf.write(fname, out_audio, 16000)
        print(f"Saved: {fname}")


if __name__ == "__main__":
    if os.path.exists("mix_4ch_tac.wav"):
        run_shift_voting("mix_4ch_tac.wav")
    else:
        print("Mix file not found.")