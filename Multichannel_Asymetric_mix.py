"""
Program to mix sources in a simulated room for FaSNet-TAC.
UPDATED: Uses asymmetric speaker positions to improve separation quality.
FIXED: RIR Analysis now handles variable lengths automatically to prevent crashes.
"""
import pyroomacoustics as pra
from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft, ifft
import os

def room_mix(files, micsetup='circular', plot=True, rt60=0.1):
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

    # Match lengths (crop to shorter)
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

    # --- KEY CHANGE: ASYMMETRIC POSITIONS ---
    # Speaker 1 (Kept same)
    room.add_source([1.5, 1.5, 1.5], signal=audio1, delay=0.0)

    # Speaker 2 (MOVED slightly to break symmetry)
    # New Position: [3.8, 3.1, 1.5] (Off-center)
    room.add_source([3.8, 3.1, 1.5], signal=audio0, delay=0.0)

    # --- MICROPHONE SETUP (Circular) ---
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
        plt.title(f"Room Setup (Asymmetric Sources)")
        plt.show()

    # Run Simulation
    print("Simulating...")
    room.simulate()

    # --- SAVING OUTPUT ---
    output_filename = "mix_4ch_tac.wav"

    # Normalize
    signal = room.mic_array.signals.T
    signal = signal / np.max(np.abs(signal)) * 0.9
    signal = (signal * 32767).astype(np.int16)

    wavfile.write(output_filename, fs0, signal)
    print(f"Saved output to: {output_filename}")

    # --- PLOT 2: RIR ANALYSIS (FIXED) ---
    if plot:
        print("Plotting RIRs...")

        # --- CALCULATION 1: Source 1 (Mic 1 vs Mic 0) ---
        rir_mic1_src0 = room.rir[1][0]
        rir_mic0_src0 = room.rir[0][0]

        # FIX: Find safe length for THIS specific pair
        n1 = min(len(rir_mic1_src0), len(rir_mic0_src0))

        # Formula: IFFT( FFT(Mic1) / FFT(Mic0) )
        numerator = fft(rir_mic1_src0[:n1])
        denominator = fft(rir_mic0_src0[:n1])

        # Add small epsilon to denominator to prevent divide-by-zero
        epsilon = 1e-10
        rrir0 = np.real(ifft(numerator / (denominator + epsilon)))

        # --- CALCULATION 2: Source 0 (Mic 0 vs Mic 1) ---
        rir_mic0_src1 = room.rir[0][1]
        rir_mic1_src1 = room.rir[1][1]

        # FIX: Find safe length for THIS specific pair
        n2 = min(len(rir_mic0_src1), len(rir_mic1_src1))

        numerator2 = fft(rir_mic0_src1[:n2])
        denominator2 = fft(rir_mic1_src1[:n2])

        rrir1 = np.real(ifft(numerator2 / (denominator2 + epsilon)))

        # --- PLOTTING GRAPHS ---
        # Plot 1
        plt.figure()
        plt.plot(rrir0)
        plt.title("Relative RIR between mic 0 and mic 1 (from Source 1)")
        plt.xlabel("Samples")
        plt.grid(True)
        plt.show()

        # Metrics for Plot 1
        maxind0 = np.argmax(np.abs(rrir0))
        print("Attenuation0", rrir0[maxind0])
        # Wrap-around check
        if maxind0 > n1/2:
            delay0 = maxind0 - n1
        else:
            delay0 = maxind0
        print("Delay0=", delay0)

        # Plot 2
        plt.figure()
        plt.plot(rrir1)
        plt.title("Relative RIR between mic 1 and mic 0 (from Source 0)")
        plt.xlabel("Samples")
        plt.grid(True)
        plt.show()

        # Metrics for Plot 2
        maxind1 = np.argmax(np.abs(rrir1))
        print("Attenuation1", rrir1[maxind1])
        if maxind1 > n2/2:
            delay1 = maxind1 - n2
        else:
            delay1 = maxind1
        print("Delay1=", delay1)

if __name__ == "__main__":
    files = ('ref1.wav', 'ref2.wav')
    room_mix(files, micsetup='circular', plot=True)