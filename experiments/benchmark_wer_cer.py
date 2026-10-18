#!/usr/bin/env python3
"""
Benchmark module for Automatic Speech Recognition (ASR).
Calculates Word Error Rate (WER) and Character Error Rate (CER) with
clinical text normalization (preserving numbers, medical dosages, and acronyms).
"""

import re
import sys
from typing import Dict, List, Tuple

def normalize_clinical_text(text: str) -> str:
    """Standardizes text for medical ASR evaluation:
    - Lowercase
    - Retain alphanumeric characters, hyphens within words, and spaces
    - Standardize common spoken punctuation words
    """
    if not text:
        return ""
    text = text.lower()
    # Normalize common spoken punctuation
    text = re.sub(r"\bperiod\b", "", text)
    text = re.sub(r"\bcomma\b", "", text)
    text = re.sub(r"\bnew paragraph\b", "", text)
    # Remove remaining punctuation except intra-word apostrophes and hyphens
    text = re.sub(r"[^\w\s\'-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def levenshtein_distance(ref: List[str], hyp: List[str]) -> Tuple[int, int, int, int]:
    """Computes Levenshtein edit distance between reference and hypothesis tokens.
    Returns: (substitutions, deletions, insertions, total_ref_tokens)
    """
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    ops = [[""] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = i
        ops[i][0] = "D"
    for j in range(m + 1):
        dp[0][j] = j
        ops[0][j] = "I"
    ops[0][0] = " "

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
                ops[i][j] = "M"
            else:
                sub = dp[i - 1][j - 1] + 1
                ins = dp[i][j - 1] + 1
                dele = dp[i - 1][j] + 1
                best = min(sub, ins, dele)
                dp[i][j] = best
                if best == sub:
                    ops[i][j] = "S"
                elif best == dele:
                    ops[i][j] = "D"
                else:
                    ops[i][j] = "I"

    # Backtrace to count S, D, I
    i, j = n, m
    subs, dels, inss = 0, 0, 0
    while i > 0 or j > 0:
        op = ops[i][j]
        if op == "M":
            i -= 1
            j -= 1
        elif op == "S":
            subs += 1
            i -= 1
            j -= 1
        elif op == "D":
            dels += 1
            i -= 1
        elif op == "I":
            inss += 1
            j -= 1
        else:
            break

    return subs, dels, inss, n

def compute_wer(ref: str, hyp: str) -> Dict[str, float]:
    ref_words = normalize_clinical_text(ref).split()
    hyp_words = normalize_clinical_text(hyp).split()
    if not ref_words:
        return {"wer": 0.0 if not hyp_words else 1.0, "sub": 0, "del": 0, "ins": len(hyp_words), "ref_len": 0}
    s, d, i, total = levenshtein_distance(ref_words, hyp_words)
    wer = (s + d + i) / total if total > 0 else 0.0
    return {
        "wer": round(wer, 4),
        "substitutions": s,
        "deletions": d,
        "insertions": i,
        "ref_len": total
    }

def compute_cer(ref: str, hyp: str) -> Dict[str, float]:
    ref_chars = list(normalize_clinical_text(ref).replace(" ", ""))
    hyp_chars = list(normalize_clinical_text(hyp).replace(" ", ""))
    if not ref_chars:
        return {"cer": 0.0 if not hyp_chars else 1.0, "ref_len": 0}
    s, d, i, total = levenshtein_distance(ref_chars, hyp_chars)
    cer = (s + d + i) / total if total > 0 else 0.0
    return {
        "cer": round(cer, 4),
        "ref_len": total
    }

# Scaffolding interface for supported models
SUPPORTED_ASR_MODELS = [
    "whisper-large-v3",
    "faster-whisper-large-v3",
    "nemo-parakeet-tdt-1.1b",
    "amazon-transcribe-medical"
]

def run_asr_evaluation(model_name: str, audio_path: str, reference_transcript: str) -> Dict[str, any]:
    """Scaffold evaluation runner. Marks unrun executions as 'Not run'."""
    if model_name not in SUPPORTED_ASR_MODELS:
        raise ValueError(f"Model {model_name} not supported.")
    
    # Placeholder for live model inference call
    return {
        "model": model_name,
        "audio_path": audio_path,
        "status": "Not run",
        "wer": None,
        "cer": None,
        "real_time_factor": None,
        "estimated_cost_usd_per_hour": {
            "whisper-large-v3": 0.006,
            "faster-whisper-large-v3": 0.002,
            "nemo-parakeet-tdt-1.1b": 0.003,
            "amazon-transcribe-medical": 1.44
        }.get(model_name, 0.0)
    }

if __name__ == "__main__":
    # Self-test unit verification
    test_ref = "Patient advised Paracetamol 650 mg TDS for three days."
    test_hyp = "Patient advise Paracetamol 650 mg TDS for 3 days"
    res_wer = compute_wer(test_ref, test_hyp)
    res_cer = compute_cer(test_ref, test_hyp)
    print("ASR Benchmark Unit Verification:")
    print("Reference:  ", test_ref)
    print("Hypothesis: ", test_hyp)
    print("WER Result: ", res_wer)
    print("CER Result: ", res_cer)
