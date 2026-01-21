"""
Program to mix sources in a PERFECTLY SYMMETRIC room for FaSNet-TAC.
Symmetry: Both speakers are exactly 1.12m away from the center mic array.
"""
import pyroomacoustics as pra
from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft, ifft
import os

def room_mix_symmetric(files, micsetup='circular', plot=True, rt60=0.1):
    # Check if files exist
    if not os.path.exists(files[0]) or not os.path.exists(files[1]):
        print(f"Error: Files not found. Looking for {files}")
        return

    # Load sources
    fs0, audio0 = wavfile.read(files[0])
    fs1, audio1 = wavfile.read(files[1])

    # Ensure Mono
    if audio0.ndim > 1: audio0 = audio0[:, 0]
    if audio1.ndim > 1: audio1 = audio1[:, 0]

    # Match lengths
    min_len = min(len(audio0), len(audio1))
    audio0 = audio0[:min_len]
    audio1 = audio1[:min_len]

    print(f"rt60={rt60}")
    room_dim = [5, 4, 2.5]  # meters

    e_absorption, max_order = pra.inverse_sabine(rt60, room_dim)
    print(f"e_absorption={e_absorption}, max_order={max_order}")

    # Create the room
    room = pra.ShoeBox(
        room_dim, fs=fs0, materials=pra.Material(e_absorption), max_order=max_order
    )

    # --- KEY CHANGE: SYMMETRIC POSITIONS ---
    # Center is [2.5, 2.0]

    # Speaker 1: [1.5, 1.5, 1.5]
    # (Offsets: -1.0 X, -0.5 Y)
    room.add_source([1.5, 1.5, 1.5], signal=audio1, delay=0.0)

    # Speaker 2: [3.5, 2.5, 1.5]
    # (Offsets: +1.0 X, +0.5 Y) -> EXACT MIRROR of Speaker 1
    room.add_source([3.5, 2.5, 1.5], signal=audio0, delay=0.0)

    # --- MICROPHONE SETUP ---
    center_x, center_y, center_z = 2.5, 2.0, 1.2

    if micsetup == 'circular':
        mic_locs = pra.circular_2D_array(center=[center_x, center_y], M=4, phi0=0, radius=0.1)
        mic_locs = np.concatenate((mic_locs, np.ones((1, 4)) * center_z), axis=0)
    elif micsetup == 'stereo':
        mic_locs = np.c_[[center_x, center_y, center_z], [center_x, center_y + 0.2, center_z]]

    room.add_microphone_array(mic_locs)

    # --- PLOT 1: ROOM SETUP ---
    if plot:
        fig, ax = room.plot()
        ax.set_xlim([0, 5])
        ax.set_ylim([0, 4])
        ax.set_zlim([0, 2.5])
        plt.title(f"Symmetric Room Setup (Equal Distances)")
        plt.show()

    # Run Simulation
    print("Simulating...")
    room.simulate()

    # --- SAVING OUTPUT ---
    output_filename = "mix_4ch_tac.wav"
    signal = room.mic_array.signals.T
    signal = signal / np.max(np.abs(signal)) * 0.9
    signal = (signal * 32767).astype(np.int16)

    wavfile.write(output_filename, fs0, signal)
    print(f"Saved symmetric mix to: {output_filename}")

    # --- PLOT 2: RIR ANALYSIS (Crash-Proof Version) ---
    if plot:
        print("Plotting RIRs...")

        # Calculation 1: Source 1
        rir_mic1_src0 = room.rir[1][0]
        rir_mic0_src0 = room.rir[0][0]
        n1 = min(len(rir_mic1_src0), len(rir_mic0_src0))

        epsilon = 1e-10
        numerator = fft(rir_mic1_src0[:n1])
        denominator = fft(rir_mic0_src0[:n1])
        rrir0 = np.real(ifft(numerator / (denominator + epsilon)))

        # Calculation 2: Source 0
        rir_mic0_src1 = room.rir[0][1]
        rir_mic1_src1 = room.rir[1][1]
        n2 = min(len(rir_mic0_src1), len(rir_mic1_src1))

        numerator2 = fft(rir_mic0_src1[:n2])
        denominator2 = fft(rir_mic1_src1[:n2])
        rrir1 = np.real(ifft(numerator2 / (denominator2 + epsilon)))

        # Plot 1
        plt.figure()
        plt.plot(rrir0)
        plt.title("Relative RIR (Source 1)")
        plt.xlabel("Samples")
        plt.grid(True)
        plt.show()

        # Plot 2
        plt.figure()
        plt.plot(rrir1)
        plt.title("Relative RIR (Source 0)")
        plt.xlabel("Samples")
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    files = ('ref1.wav', 'ref2.wav')
    room_mix_symmetric(files, micsetup='circular', plot=True)