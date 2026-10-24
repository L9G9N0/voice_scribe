#!/usr/bin/env python3
"""
BTP Voice Scribe: ClinicalMix Controlled Audio Simulation Script
Simulates realistic outpatient clinic (OPD) acoustic conditions by combining:
1. Target clinical consultation (e.g. PriMock57 isolated channels)
2. Competing background speech (e.g. secondary dialogues from Svarah/MeDial-Speech)
3. Ambient clinic/environmental noise (e.g. DEMAND office/hallway or hospital soundscapes)

Ground-Truth Principle:
The reference transcript corresponds STRICTLY to the target consultation.
Any transcription of background distractor speech is evaluated as an insertion error.
"""

import argparse
import json
import os
import sys
import numpy as np

def compute_active_rms(audio: np.ndarray, threshold_ratio: float = 0.05) -> float:
    """Computes RMS of active speech frames above an energy threshold."""
    frame_len = 512
    frames = [audio[i:i+frame_len] for i in range(0, len(audio) - frame_len, frame_len)]
    frame_rms = [np.sqrt(np.mean(f**2)) for f in frames]
    peak_rms = max(frame_rms) if frame_rms else 1e-6
    active = [rms for rms in frame_rms if rms > threshold_ratio * peak_rms]
    return float(np.mean(active)) if active else float(np.sqrt(np.mean(audio**2)))

def scale_audio_to_ratio(target: np.ndarray, interference: np.ndarray, ratio_db: float) -> np.ndarray:
    """
    Scales interference audio so that 10 * log10(P_target / P_interference) = ratio_db.
    """
    p_target = max(compute_active_rms(target)**2, 1e-12)
    p_interf = max(compute_active_rms(interference)**2, 1e-12)
    desired_p_interf = p_target / (10 ** (ratio_db / 10.0))
    scale_factor = np.sqrt(desired_p_interf / p_interf)
    return interference * scale_factor

def mix_clinical_audio(
    target_audio: np.ndarray,
    distractor_audio: np.ndarray = None,
    noise_audio: np.ndarray = None,
    sir_db: float = 10.0,
    snr_db: float = 15.0
) -> np.ndarray:
    """Mixes target consultation with scaled background speech (SIR) and ambient noise (SNR)."""
    n_samples = len(target_audio)
    mixture = target_audio.copy()

    if distractor_audio is not None and len(distractor_audio) > 0:
        # Loop or slice distractor to match length
        if len(distractor_audio) < n_samples:
            repeats = int(np.ceil(n_samples / len(distractor_audio)))
            distractor_audio = np.tile(distractor_audio, repeats)[:n_samples]
        else:
            distractor_audio = distractor_audio[:n_samples]

        scaled_distractor = scale_audio_to_ratio(target_audio, distractor_audio, sir_db)
        mixture += scaled_distractor

    if noise_audio is not None and len(noise_audio) > 0:
        if len(noise_audio) < n_samples:
            repeats = int(np.ceil(n_samples / len(noise_audio)))
            noise_audio = np.tile(noise_audio, repeats)[:n_samples]
        else:
            noise_audio = noise_audio[:n_samples]

        scaled_noise = scale_audio_to_ratio(target_audio, noise_audio, snr_db)
        mixture += scaled_noise

    # Peak normalization to prevent clipping
    max_val = np.max(np.abs(mixture))
    if max_val > 0.99:
        mixture = mixture / max_val * 0.95

    return mixture

def main():
    parser = argparse.ArgumentParser(description="ClinicalMix: Controlled Clinic Audio Simulator")
    parser.add_argument("--doctor_wav", required=True, help="Path to clean doctor channel WAV")
    parser.add_argument("--patient_wav", required=True, help="Path to clean patient channel WAV")
    parser.add_argument("--distractor_wav", default=None, help="Path to competing background speech WAV")
    parser.add_argument("--noise_wav", default=None, help="Path to ambient clinic/room noise WAV")
    parser.add_argument("--sir_db", type=float, default=10.0, help="Signal-to-Interference Ratio (dB) [default: 10.0]")
    parser.add_argument("--snr_db", type=float, default=15.0, help="Signal-to-Noise Ratio (dB) [default: 15.0]")
    parser.add_argument("--output_wav", required=True, help="Output path for mixed simulation WAV")
    parser.add_argument("--output_json", default=None, help="Path for simulation metadata JSON")

    args = parser.parse_args()

    try:
        import soundfile as sf
    except ImportError:
        sys.exit("Error: 'soundfile' library required. Install via pip install soundfile")

    doc_data, sr_doc = sf.read(args.doctor_wav)
    pat_data, sr_pat = sf.read(args.patient_wav)
    assert sr_doc == sr_pat, f"Sample rate mismatch: {sr_doc} vs {sr_pat}"
    sr = sr_doc

    min_len = min(len(doc_data), len(pat_data))
    target_clean = doc_data[:min_len] + pat_data[:min_len]

    distractor_data = None
    if args.distractor_wav and os.path.exists(args.distractor_wav):
        distractor_data, sr_dist = sf.read(args.distractor_wav)
        if sr_dist != sr:
            sys.exit(f"Sample rate mismatch: distractor ({sr_dist}) != target ({sr})")

    noise_data = None
    if args.noise_wav and os.path.exists(args.noise_wav):
        noise_data, sr_noise = sf.read(args.noise_wav)
        if sr_noise != sr:
            sys.exit(f"Sample rate mismatch: noise ({sr_noise}) != target ({sr})")

    mixed = mix_clinical_audio(
        target_audio=target_clean,
        distractor_audio=distractor_data,
        noise_audio=noise_data,
        sir_db=args.sir_db,
        snr_db=args.snr_db
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.output_wav)), exist_ok=True)
    sf.write(args.output_wav, mixed, sr)
    print(f"Successfully generated ClinicalMix simulation: {args.output_wav} ({len(mixed)/sr:.2f}s)")

    if args.output_json:
        metadata = {
            "doctor_source": args.doctor_wav,
            "patient_source": args.patient_wav,
            "distractor_source": args.distractor_wav,
            "noise_source": args.noise_wav,
            "sir_db": args.sir_db,
            "snr_db": args.snr_db,
            "sample_rate_hz": sr,
            "duration_seconds": len(mixed) / sr,
            "ground_truth_target_only": True
        }
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved metadata: {args.output_json}")

if __name__ == "__main__":
    main()
