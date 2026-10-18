# BTP Voice Scribe: Benchmark Scaffolding & Evaluation Protocol

## 1. Evaluation Protocol & Metrics
All models in the Voice Scribe pipeline are evaluated under standardized, reproducible benchmark protocols:

1. **Speech-to-Text (ASR)**:
   - **Word Error Rate (WER)**: Standard Levenshtein edit distance on normalized word sequences: $\text{WER} = \frac{S + D + I}{N_{\text{ref}}}$.
   - **Character Error Rate (CER)**: Levenshtein distance on character sequences without spaces.
   - **Medical Entity Recall (M-Recall)**: Percentage of clinically relevant terms (drugs, dosages, anatomical sites, diagnoses) correctly transcribed.
   - **Real-Time Factor (RTF)**: $\frac{\text{Processing Time (seconds)}}{\text{Audio Duration (seconds)}}$. Values $<1.0$ indicate faster-than-real-time.
2. **Speaker Diarization**:
   - **Diarization Error Rate (DER)**: Total percentage of diarization error with a standard 250ms collar: $\text{DER} = \frac{\text{MISS} + \text{FA} + \text{CONF}}{\text{Total Speech Time}}$.
   - **Jaccard Error Rate (JER)**: Average across all speakers of the Jaccard distance between reference and hypothesis speaking intervals.
3. **Doctor vs. Patient Role Attribution**:
   - **Attribution Accuracy & F1-Score**: Binary classification performance determining whether speaker turns belong to Doctor or Patient.
4. **Estimated Cost**:
   - USD per hour of processed audio on cloud infrastructure (A10G GPU / Amazon Transcribe Medical API pricing).

---

## 2. Benchmark Scaffolding Matrix

Per Hard Rule 7, all models not yet executed on local GPU clusters are explicitly labeled as **`Not run`**.

| Pipeline Stage | Model Identifier | Target Benchmark Dataset | WER (%) | CER (%) | DER (%) | Attribution F1 | RTF | Est. Cost ($/hr) | Status |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ASR** | `whisper-large-v3` | `primock57` (Doctor/Patient) | — | — | — | — | — | $0.006 | **Not run** |
| **ASR** | `faster-whisper-large-v3` | `eka_medical_asr` (Indian Clinical) | — | — | — | — | — | $0.002 | **Not run** |
| **ASR** | `nemo-parakeet-tdt-1.1b` | `modulate_synthetic_dictation` | — | — | — | — | — | $0.003 | **Not run** |
| **ASR** | `amazon-transcribe-medical` | `primock57` | — | — | — | — | — | $1.440 | **Not run** |
| **Diarization** | `pyannote/speaker-diarization-3.1` | `ami_corpus` (Headset Mix) | — | — | — | — | — | $0.005 | **Not run** |
| **Diarization** | `nemo/diar_msdd_telephonic` | `voxconverse` | — | — | — | — | — | $0.007 | **Not run** |
| **Diarization** | `faster-whisper-diarization` | `primock57` (Mixed Channels) | — | — | — | — | — | $0.003 | **Not run** |
| **Attribution** | `clinical-llm-role-prompting` | `primock57` Transcripts | — | — | — | — | — | $0.001 | **Not run** |
| **Attribution** | `acoustic-voice-profile-classifier` | `medial_speech` | — | — | — | — | — | $0.001 | **Not run** |

---

## 3. How to Execute Live Benchmarks
Once GPU resources and model weights are downloaded, execute:
```bash
# Run benchmark scaffolding over template manifest
python3 experiments/run_benchmarks.py research/audio-datasets/MANIFEST_TEMPLATE.csv
```
