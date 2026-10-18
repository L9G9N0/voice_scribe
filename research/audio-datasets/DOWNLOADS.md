# BTP Voice Scribe: Dataset Downloads & Storage Ledger

## 1. Local Storage Policy & Hard Rules
1. **Zero Raw Audio in Git**: Per Hard Rule 6, no `.wav`, `.mp3`, `.flac`, `.opus`, archives, or raw audio datasets are stored in the git repository. All media files are blocked by `.gitignore`.
2. **External Storage Staging**: Audio datasets and evaluation splits are downloaded to an isolated external storage path (e.g. `/tmp/voice_scribe_eval_cache` or `/Volumes/ExternalStorage/voice_scribe_data`).
3. **Integrity Verification**: Every downloaded asset is verified against its published SHA-256 / MD5 checksum before being admitted to the benchmark pipeline.

---

## 2. Download Log & Verification Status

| Dataset ID | Asset Name | Storage Location | Size (Bytes) | SHA-256 Checksum | Verified? |
| :--- | :--- | :--- | :--- | :--- | :---: |
| `primock57` | `day1_consultation01_doctor.wav` | `/tmp/voice_scribe_eval_cache/day1_consultation01_doctor.wav` | 14,653,484 | `a93ca69f49da821b2d10b03f3189fd0d82200a697805c18f316719ba328ff96b` | **PASS (Exact Match)** |
| `medial_speech` | `icassp2026-benchmark.zip` | Streamed / Inspected in-memory | 488,843 | `4c8d5a165a2012678fb485bb2155799a4c8f58c73cbb7b919d701dfc333a4ff8` | **PASS (Valid ZIP)** |
| `eka_medical_asr` | `en/test-00000.parquet` | Inspected via Hugging Face API | 1,809,500,000 | Verified via HF commit hash `8c00539` | **PASS (Valid Parquet)** |
| `synth_dopaco` | `data/dev-00000.tar` | Streamed / Inspected via WebDataset API | 469,422,080 | Verified via HF commit hash `c40a5a2` | **PASS (Valid Shard)** |

---

## 3. Storage Footprint Estimates for Full Corpus Ingestion

| Dataset ID | Target Split | Format | Est. Uncompressed Storage | Est. Compressed Download | Recommended Action |
| :--- | :--- | :--- | :---: | :---: | :--- |
| `primock57` | Complete (57 visits) | WAV 16kHz 16-bit | 1.8 GB | 1.2 GB | **Full local clone** via `git lfs` |
| `medial_speech` | Complete (Part 1 + 2) | WAV 16kHz | 15.0 GB | 9.26 GB | **Full download** to external drive |
| `eka_medical_asr` | English Test (`en`) | Parquet (embedded WAV) | 1.81 GB | 1.54 GB | **Cache locally** via `datasets` |
| `synth_dopaco` | Dev split (400 visits) | Opus TAR shard | 0.5 GB | 0.47 GB | **Download dev shard**; stream train |
| `modulate_synthetic_dictation`| Quick subset (120 notes) | WAV 16kHz | 0.4 GB | 0.25 GB | **Download quick subset** |
| `ami_corpus` | Headset audio subset | WAV 16kHz | 12.0 GB | 8.5 GB | **Download core test set** (16 meetings) |
| `voxconverse` | Dev split audio | WAV 16kHz | 10.0 GB | 8.0 GB | **Download dev split** |

---

## 4. Failed Downloads, Blockers & Access Restrictions

| Dataset ID | Action Attempted | Result / Status | Official Remedy / Next Step |
| :--- | :--- | :--- | :--- |
| `whissleai_simulated_exams` | Download README / Parquet | **HTTP 401 Restricted** | Gated on Hugging Face; request organizational access via Hugging Face UI. |
| `physionet_egd_cxr` | Direct Wget of audio files | **Gated (Credentialed)** | Requires human research training (CITI certificate) and signed DUA on PhysioNet. |
| `dihard3_clinical` | LDC direct download | **Gated (LDC Membership)** | Requires university institutional membership or license purchase from LDC catalog. |
| `defined_ai_doctor_patient`| Commercial catalog access | **Commercial Purchase** | Requires commercial contract with Defined.ai enterprise sales. |
| `paultimothymooney_kaggle`| Direct Kaggle URL fetch | **HTTP 404 (Auth Needed)** | Resolved via verified open Hugging Face mirror (`Shamus/Medical_Speech_Transcription_and_Intent`). |
