import numpy as np
import soundfile as sf
import mir_eval


def verify_metrics():
    print("--- Verifying Metric Assignment ---")

    # 1. Load Data
    ref1, fs = sf.read("ref1.wav")
    ref2, _ = sf.read("ref2.wav")
    est1, _ = sf.read("fasnet_final_1.wav")
    est2, _ = sf.read("fasnet_final_2.wav")

    # 2. Force Mono & Crop
    def prep(d):
        return d[:min_len, 0] if d.ndim > 1 else d[:min_len]

    min_len = min(len(ref1), len(est1))

    refs = np.stack([prep(ref1), prep(ref2)])
    ests = np.stack([prep(est1), prep(est2)])

    # 3. Calculate
    # The function returns: sdr, sir, sar, perm
    metrics = mir_eval.separation.bss_eval_sources(refs, ests, compute_permutation=True)

    sdr_vals = metrics[0]
    sir_vals = metrics[1]
    sar_vals = metrics[2]

    print("\nOFFICIAL MIR_EVAL OUTPUT MAPPING:")
    print(f"Index 0 (SDR) : {np.mean(sdr_vals):.2f} dB  -> (Overall Quality)")
    print(f"Index 1 (SIR) : {np.mean(sir_vals):.2f} dB  -> (Separation / Bleed)")
    print(f"Index 2 (SAR) : {np.mean(sar_vals):.2f} dB  -> (Artifacts / Robotic Sound)")
    print("-" * 50)

    if np.mean(sir_vals) < 5.0:
        print("⚠️ DIAGNOSIS: SIR is very low.")
        print("   This means the speakers are NOT separated. They are still mixed together.")
    elif np.mean(sar_vals) < 5.0:
        print("⚠️ DIAGNOSIS: SAR is very low.")
        print("   This means the audio is separated but sounds destroyed/robotic.")
    else:
        print("✅ DIAGNOSIS: Metrics look distinct and valid.")


if __name__ == "__main__":
    verify_metrics()