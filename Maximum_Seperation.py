"""
Program to mix sources for OPTIMAL FaSNet-TAC Performance.
Strategy:
1. Moderate Reverb (RT60=0.1) -> Gives model the spatial cues it needs.
2. Asymmetric Positions -> Breaks symmetry confusion.
3. Wide Angles -> Maximum separation without being "unnatural".
"""
import pyroomacoustics as pra
from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np
import os

def room_mix_best(files, plot=True):
    print("--- Generating Balanced Asymmetric Mix ---")

    # 1. Load Files
    if not os.path.exists(files[0]) or not os.path.exists(files[1]):
        print(f"Error: Files not found. Looking for {files}")
        return

    fs0, audio0 = wavfile.read(files[0])
    fs1, audio1 = wavfile.read(files[1])

    if audio0.ndim > 1: audio0 = audio0[:, 0]
    if audio1.ndim > 1: audio1 = audio1[:, 0]

    min_len = min(len(audio0), len(audio1))
    audio0, audio1 = audio0[:min_len], audio1[:min_len]

    # --- SETTING 1: REALISTIC REVERB ---
    # We go back to RT60=0.1. The AI *likes* this small amount of reverb
    # because it helps define the "space" (Depth Perception).
    rt60 = 0.1
    room_dim = [5, 4, 2.5]
    e_absorption, max_order = pra.inverse_sabine(rt60, room_dim)

    room = pra.ShoeBox(
        room_dim, fs=fs0, materials=pra.Material(e_absorption), max_order=max_order
    )

    # --- SETTING 2: MICROPHONE ARRAY ---
    center_x, center_y, center_z = 2.5, 2.0, 1.2
    mic_locs = pra.circular_2D_array(center=[center_x, center_y], M=4, phi0=0, radius=0.1)
    mic_locs = np.concatenate((mic_locs, np.ones((1, 4)) * center_z), axis=0)
    room.add_microphone_array(mic_locs)

    # --- SETTING 3: ASYMMETRIC POSITIONS (The Winner) ---
    # We use the positions that gave us the 9dB result,
    # but push them slightly further apart for clarity.

    # Speaker 1 (Left-ish, slightly forward)
    room.add_source([1.5, 1.8, 1.5], signal=audio1, delay=0.0)

    # Speaker 2 (Right-ish, back, creating asymmetry)
    room.add_source([3.8, 3.1, 1.5], signal=audio0, delay=0.0)

    # --- PLOT ---
    if plot:
        fig, ax = room.plot()
        ax.set_xlim([0, 5])
        ax.set_ylim([0, 4])
        plt.title("Balanced Mix (Asymmetric + Reverb)")
        plt.show()

    # Simulate
    print("Simulating...")
    room.simulate()

    # Save
    output_filename = "mix_4ch_tac.wav"
    signal = room.mic_array.signals.T

    # Normalize
    max_val = np.max(np.abs(signal))
    if max_val > 0: signal = signal / max_val * 0.9
    signal = (signal * 32767).astype(np.int16)

    wavfile.write(output_filename, fs0, signal)
    print(f"Saved best mix t: {output_filename}")

if __name__ == "__main__":
    files = ('ref1.wav', 'ref2.wav') 
    room_mix_best(files, plot=True)