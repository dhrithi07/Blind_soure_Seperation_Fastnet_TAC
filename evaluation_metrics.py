import numpy as np
import soundfile as sf
import mir_eval
import os


def load_files(file_list):
    """Loads a list of files and returns a list of numpy arrays."""
    loaded_data = []
    for f in file_list:
        data, fs = sf.read(f)
        # Ensure mono
        if data.ndim > 1:
            data = data[:, 0]
        loaded_data.append(data)
    return loaded_data


def evaluate():
    print("--- Evaluating Results: Raw vs. Soft Mask vs. Hybrid Mask ---")

    # 1. Define File Groups
    ref_files = ['ref1.wav', 'ref2.wav']
    mix_file = 'mix_4ch_tac.wav'

    # Group A: Raw Model Output
    raw_files = ['fasnet_output_1.wav', 'fasnet_output_2.wav']

    # Group B: Soft Mask Output (Original Post-Process)
    # Ensure you ran post_process.py (soft mask) to generate these
    soft_files = ['fasnet_masked_1.wav', 'fasnet_masked_2.wav']

    # Group C: Hybrid/Hard Output (Current Post-Process)
    # Ensure you ran post_process_hybrid.py to generate these
    hybrid_files = ['fasnet_mvdr_hybrid_1.wav', 'fasnet_mvdr_hybrid_2.wav']

    # 2. Check Files
    required = ref_files + [mix_file] + raw_files
    if not all(os.path.exists(f) for f in required):
        print(f"Error: Missing core files. Please ensure {required} exist.")
        return

    has_soft = all(os.path.exists(f) for f in soft_files)
    has_hybrid = all(os.path.exists(f) for f in hybrid_files)

    if not has_soft:
        print("Warning: 'fasnet_masked' (Soft) files not found. Skipping Soft column.")
    if not has_hybrid:
        print("Warning: 'fasnet_hybrid' files not found. Skipping Hybrid column.")

    # 3. Load Data
    refs_data = load_files(ref_files)
    mix_data, _ = sf.read(mix_file)
    raw_data = load_files(raw_files)

    soft_data = []
    if has_soft:
        soft_data = load_files(soft_files)

    hybrid_data = []
    if has_hybrid:
        hybrid_data = load_files(hybrid_files)

    # 4. Find GLOBAL Minimum Length
    lengths = [len(r) for r in refs_data] + \
              [len(r) for r in raw_data] + \
              [mix_data.shape[0]]

    if has_soft:
        lengths += [len(s) for s in soft_data]
    if has_hybrid:
        lengths += [len(h) for h in hybrid_data]

    min_len = min(lengths)
    print(f"Trimming evaluation to {min_len} samples ({min_len / 16000:.2f} seconds)...")

    # 5. Crop and Stack Arrays
    ref_stack = np.stack([r[:min_len] for r in refs_data])

    # Mix Baseline (Crop and duplicate Ch0)
    mix_mono = mix_data[:min_len, 0]
    mix_stack = np.stack([mix_mono, mix_mono])

    # Raw Stack
    raw_stack = np.stack([r[:min_len] for r in raw_data])

    # Soft Stack
    soft_stack = None
    if has_soft:
        soft_stack = np.stack([s[:min_len] for s in soft_data])

    # Hybrid Stack
    hybrid_stack = None
    if has_hybrid:
        hybrid_stack = np.stack([h[:min_len] for h in hybrid_data])

    # 6. Calculate Metrics
    print("\nCalculating metrics (this usually takes a few seconds)...")

    # A. Baseline
    sdr_b, sir_b, sar_b, _ = mir_eval.separation.bss_eval_sources(
        ref_stack, mix_stack, compute_permutation=True
    )

    # B. Raw Model
    sdr_r, sir_r, sar_r, _ = mir_eval.separation.bss_eval_sources(
        ref_stack, raw_stack, compute_permutation=True
    )

    # C. Soft Mask
    sdr_s, sir_s, sar_s = None, None, None
    if has_soft:
        sdr_s, sir_s, sar_s, _ = mir_eval.separation.bss_eval_sources(
            ref_stack, soft_stack, compute_permutation=True
        )

    # D. Hybrid Mask
    sdr_h, sir_h, sar_h = None, None, None
    if has_hybrid:
        sdr_h, sir_h, sar_h, _ = mir_eval.separation.bss_eval_sources(
            ref_stack, hybrid_stack, compute_permutation=True
        )

    # 7. Print Consolidated Table
    print("\n" + "=" * 115)
    # Adjust header formatting for the extra column
    header = f"{'Metric':<8} | {'Mix':<10} | {'Raw (FaSNet)':<15} | {'Soft Mask':<15} | {'Hybrid Mask':<15} | {'Best Gain':<10}"
    print(header)
    print("-" * 115)

    def print_row(name, base, raw, soft, hybrid):
        avg_b = np.mean(base)
        avg_r = np.mean(raw)

        # Prepare values for printing
        val_soft = np.mean(soft) if soft is not None else 0.0
        str_soft = f"{val_soft:<15.2f}" if soft is not None else f"{'N/A':<15}"

        val_hybrid = np.mean(hybrid) if hybrid is not None else 0.0
        str_hybrid = f"{val_hybrid:<15.2f}" if hybrid is not None else f"{'N/A':<15}"

        # Calculate max gain (Hybrid vs Mix)
        # You can change this to compare vs Raw if preferred
        current_best = avg_r
        if soft is not None: current_best = max(current_best, val_soft)
        if hybrid is not None: current_best = max(current_best, val_hybrid)

        gain = current_best - avg_b

        print(f"{name:<8} | {avg_b:<10.2f} | {avg_r:<15.2f} | {str_soft} | {str_hybrid} | +{gain:.2f} dB")

    print_row("SDR", sdr_b, sdr_r, sdr_s, sdr_h)
    print_row("SIR", sir_b, sir_r, sir_s, sir_h)
    print_row("SAR", sar_b, sar_r, sar_s, sar_h)
    print("-" * 115)

    if has_hybrid and has_soft:
        print("\nComparison Note:")
        print("* Soft Mask: Better SAR (Smoother audio, fewer artifacts)")
        print("* Hybrid Mask: Better SIR (Better separation, less noise)")


if __name__ == "__main__":
    evaluate()