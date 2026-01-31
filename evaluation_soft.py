import numpy as np
import soundfile as sf
import mir_eval

def verify_metrics():
    print("--- Verifying Metric Assignment ---")

    # 1. Load Data
    ref1, fs = sf.read("ref1_16k.wav")
    ref2, _ = sf.read("ref2_16k.wav")
    est1, _ = sf.read("fasnet_final_1.wav")
    est2, _ = sf.read("fasnet_final_2.wav")

    # 2. Find GLOBAL Minimum Length
    # You must check all 4 files to ensure they can be stacked
    min_len = min(len(ref1), len(ref2), len(est1), len(est2))
    print(f"Syncing all files to {min_len} samples...")

    def prep(d):
        # Crop to min_len and ensure mono
        d_cropped = d[:min_len]
        return d_cropped[:, 0] if d_cropped.ndim > 1 else d_cropped

    # Now all arrays are guaranteed to be exactly (min_len,)
    refs = np.stack([prep(ref1), prep(ref2)])
    ests = np.stack([prep(est1), prep(est2)])

    # 3. Calculate
    # mir_eval expects (n_sources, n_samples)
    metrics = mir_eval.separation.bss_eval_sources(refs, ests, compute_permutation=True)

    sdr_vals, sir_vals, sar_vals, _ = metrics

    print("\nOFFICIAL MIR_EVAL OUTPUT MAPPING:")
    print(f"Index 0 (SDR) : {np.mean(sdr_vals):.2f} dB  -> (Overall Quality)")
    print(f"Index 1 (SIR) : {np.mean(sir_vals):.2f} dB  -> (Separation / Bleed)")
    print(f"Index 2 (SAR) : {np.mean(sar_vals):.2f} dB  -> (Artifacts / Robotic Sound)")
    print("-" * 50)

    if np.mean(sir_vals) < 5.0:
        print("DIAGNOSIS: SIR is very low.")
        print("   This means the speakers are NOT separated. They are still mixed together.")
    elif np.mean(sar_vals) < 5.0:
        print(" DIAGNOSIS: SAR is very low.")
        print("   This means the audio is separated but sounds destroyed/robotic.")
    else:
        print( "DIAGNOSIS: Metrics look distinct and valid.")


if __name__ == "__main__":
    verify_metrics()
