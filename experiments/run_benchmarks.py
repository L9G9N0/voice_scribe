#!/usr/bin/env python3
"""
Orchestration script for running BTP Voice Scribe evaluation benchmarks.
Runs ASR (WER/CER), Diarization (DER), and Role Attribution benchmarks
across manifest files, tracking Real-Time Factor (RTF) and inference costs.
Unrun models are explicitly recorded as 'Not run'.
"""

import csv
import json
import os
import sys
import time
from typing import Dict, List

from benchmark_wer_cer import compute_wer, compute_cer, SUPPORTED_ASR_MODELS
from benchmark_diarization import SUPPORTED_DIARIZATION_MODELS
from benchmark_attribution import compute_attribution_metrics, SUPPORTED_ATTRIBUTION_MODELS

def evaluate_manifest(manifest_csv: str) -> Dict[str, any]:
    if not os.path.exists(manifest_csv):
        print(f"Manifest {manifest_csv} not found.", file=sys.stderr)
        return {"error": "Manifest not found"}

    rows = []
    with open(manifest_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    print(f"Loaded {len(rows)} samples from {manifest_csv}")

    results = {
        "manifest": manifest_csv,
        "sample_count": len(rows),
        "asr_benchmarks": {},
        "diarization_benchmarks": {},
        "attribution_benchmarks": {}
    }

    # Scaffold ASR models
    for model in SUPPORTED_ASR_MODELS:
        results["asr_benchmarks"][model] = {
            "status": "Not run",
            "wer": None,
            "cer": None,
            "rtf": None,
            "cost_per_hour_usd": {
                "whisper-large-v3": 0.006,
                "faster-whisper-large-v3": 0.002,
                "nemo-parakeet-tdt-1.1b": 0.003,
                "amazon-transcribe-medical": 1.44
            }.get(model, 0.0)
        }

    # Scaffold Diarization models
    for model in SUPPORTED_DIARIZATION_MODELS:
        results["diarization_benchmarks"][model] = {
            "status": "Not run",
            "der": None,
            "jer": None,
            "rtf": None,
            "cost_per_hour_usd": {
                "pyannote/speaker-diarization-3.1": 0.005,
                "nemo/diar_msdd_telephonic": 0.007,
                "faster-whisper-diarization": 0.003
            }.get(model, 0.0)
        }

    # Scaffold Attribution models
    for model in SUPPORTED_ATTRIBUTION_MODELS:
        results["attribution_benchmarks"][model] = {
            "status": "Not run",
            "accuracy": None,
            "f1_score": None
        }

    return results

if __name__ == "__main__":
    manifest = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(__file__), "..", "research", "audio-datasets", "MANIFEST_TEMPLATE.csv"
    )
    report = evaluate_manifest(manifest)
    print("\nBenchmark Scaffolding Status Report:")
    print(json.dumps(report, indent=2))
