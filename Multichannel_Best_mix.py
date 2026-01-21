import numpy as np
import soundfile as sf
import pyroomacoustics as pra
import os


def load_safe_mono(filename):
    """Loads audio and guarantees it is a 1D float array."""
    data, fs = sf.read(filename)

    # If Stereo or 2D column vector, take the first channel
    if data.ndim > 1:
        data = data[:, 0]

    # Ensure it's 1D (flatten) and float type
    return data.flatten().astype(np.float32)


def create_easy_mix():
    print("--- Generating 'Golden Setup' Mix (High SAR / High SIR) ---")

    # 1. Configuration
    audio_files = ["ref1.wav", "ref2.wav"]
    room_dim = [6, 5, 3]  # Meters

    # RT60 = 0.15s (Very dry room = Easier for AI = Higher SAR)
    try:
        e_absorption, max_order = pra.inverse_sabine(0.15, room_dim)
    except:
        # Fallback if sabine fails
        e_absorption = 0.3
        max_order = 10

    # 2. Setup Room
    room = pra.ShoeBox(
        room_dim,
        fs=16000,
        materials=pra.Material(e_absorption),
        max_order=max_order
    )

    # 3. Microphone Array (Center of room)
    center = [3.0, 2.5, 1.2]
    R = 0.10
    mic_locs = np.c_[
        [center[0] + R, center[1], center[2]],  # Mic 1
        [center[0], center[1] + R, center[2]],  # Mic 2
        [center[0] - R, center[1], center[2]],  # Mic 3
        [center[0], center[1] - R, center[2]],  # Mic 4
    ]
    room.add_microphone_array(mic_locs)

    # 4. Add Sources (Using the Safe Loader)
    # Source 1: 45 degrees
    src1_audio = load_safe_mono(audio_files[0])
    room.add_source([center[0] + 1.0, center[1] + 1.0, 1.2], signal=src1_audio)

    # Source 2: 135 degrees
    src2_audio = load_safe_mono(audio_files[1])
    room.add_source([center[0] - 1.0, center[1] + 1.0, 1.2], signal=src2_audio)

    # 5. Simulate
    print("Simulating acoustics (Ray Tracing)...")
    room.simulate()

    # 6. Save Mix
    mix_audio = room.mic_array.signals.T
    # Normalize
    mix_audio = mix_audio / np.max(np.abs(mix_audio)) * 0.9
    sf.write("mix_4ch_tac.wav", mix_audio, 16000)
    print("✅ Saved: mix_4ch_tac.wav")

    # 7. Generate Clean References (Crucial for Evaluation)
    print("Generating Clean References (with same room acoustics)...")

    # Ref 1 Context
    room_ref1 = pra.ShoeBox(room_dim, fs=16000, materials=pra.Material(e_absorption), max_order=max_order)
    room_ref1.add_microphone_array(mic_locs)
    room_ref1.add_source([center[0] + 1.0, center[1] + 1.0, 1.2], signal=src1_audio)
    room_ref1.simulate()
    ref1_out = room_ref1.mic_array.signals[0, :]
    sf.write("ref1_ideal.wav", ref1_out / np.max(np.abs(ref1_out)) * 0.9, 16000)

    # Ref 2 Context
    room_ref2 = pra.ShoeBox(room_dim, fs=16000, materials=pra.Material(e_absorption), max_order=max_order)
    room_ref2.add_microphone_array(mic_locs)
    room_ref2.add_source([center[0] - 1.0, center[1] + 1.0, 1.2], signal=src2_audio)
    room_ref2.simulate()
    ref2_out = room_ref2.mic_array.signals[0, :]
    sf.write("ref2_ideal.wav", ref2_out / np.max(np.abs(ref2_out)) * 0.9, 16000)

    print("✅ Saved: ref1_ideal.wav & ref2_ideal.wav")


if __name__ == "__main__":
    if os.path.exists("ref1.wav") and os.path.exists("ref2.wav"):
        create_easy_mix()
    else:
        print("Error: 'ref1.wav' and 'ref2.wav' not found.")