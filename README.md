# RTCC — Real-Time Collaborative Cloner

> A minimal, fully local command-line tool for zero-shot voice cloning.  
> Powered by [OpenVoice](https://github.com/myshell-ai/OpenVoice) by MyShell AI & MIT.

---

## Table of Contents

1. [What Is RTCC?](#1-what-is-rtcc)
2. [Architecture](#2-architecture)
3. [Installation](#3-installation)
4. [How to Use](#4-how-to-use)
5. [CLI Reference](#5-cli-reference)
6. [Troubleshooting](#6-troubleshooting)
7. [Hardware Requirements](#7-hardware-requirements)
8. [Project Structure](#8-project-structure)
9. [Dependencies](#9-dependencies)
10. [Phase 1 — Usable Engine](#10-phase-1--usable-engine)

---

## 1. What Is RTCC?

RTCC (Real-Time Collaborative Cloner) is a minimal, fully local command-line tool for voice cloning.
It takes a short reference audio sample of any speaker and can:

- **Generate speech from text** in the cloned voice _(Text → Voice)_
- **Convert any audio recording** into the cloned voice _(Audio → Voice Conversion)_

The system is built on **OpenVoice V2**, an open-source zero-shot voice cloning model developed by MIT and MyShell AI.
There is no server, no web UI, no API — everything runs locally on your machine.

### Key Characteristics

| Feature | Detail |
|---------|--------|
| Zero-shot cloning | No fine-tuning required — just a 3–10 second reference clip |
| Fully offline | Runs entirely on CPU or a single GPU |
| Two modes | Text synthesis and audio voice conversion |
| Language | English out of the box |
| Output | Standard `.wav` file |

---

## 2. Architecture

RTCC wraps OpenVoice through a clean 4-file separation of concerns:

| File | Role |
|------|------|
| `rtcc_cli.py` | CLI entry point — argument parsing, path validation, pipeline dispatch |
| `voice_cloner.py` | `VoiceCloner` class — orchestrates TTS, SE extraction, tone conversion |
| `openvoice_loader.py` | Model loader — instantiates `BaseSpeakerTTS` & `ToneColorConverter` |
| `audio_utils.py` | Audio I/O — load, resample, save WAV files via torchaudio / soundfile |

### Text → Voice Pipeline

```
Text Input
    │
    ▼
BaseSpeakerTTS.tts()          →  base_speech.wav  (neutral voice)
    │
    ▼
ToneColorConverter.extract_se(base_speech)  →  src_se  (source embedding)
    │
    ▼
ToneColorConverter.convert(src_se, tgt_se=reference_se)
    │
    ▼
output.wav  (cloned voice)
```

### Audio → Voice Conversion Pipeline

```
input_audio.wav
    │
    ▼
ToneColorConverter.extract_se(input_audio)  →  src_se  (source embedding)
    │
    ▼
reference_audio.wav  →  extract_se()  →  tgt_se  (target embedding)
    │
    ▼
ToneColorConverter.convert(src_se, tgt_se)
    │
    ▼
output.wav  (reference speaker's voice, original content)
```

### How Speaker Embedding Works

OpenVoice encodes vocal timbre (tone colour) into a compact vector called a **Speaker Embedding (SE)**.
The `ToneColorConverter` blends the _content_ of one audio stream with the _timbre vector_ of the reference speaker —
producing speech that sounds like the reference speaker, regardless of what they are saying.

---

## 3. Installation

### Prerequisites

- Python **3.11** — 3.12+ is NOT compatible with OpenVoice's dependency stack
- `pip`, `git`
- ~2 GB disk space for model checkpoints
- Optional: CUDA-capable GPU for faster inference

### Step 1 — Set up Python 3.11 environment

```bash
brew install pyenv
pyenv install 3.11.9
pyenv local 3.11.9

python -m venv rtcc-venv
source rtcc-venv/bin/activate

python --version   # should print Python 3.11.x
```

### Step 2 — Clone OpenVoice

```bash
git clone https://github.com/myshell-ai/OpenVoice
```

### Step 3 — Install dependencies

```bash
# Install numpy first to avoid version pin conflict
pip install "numpy>=1.23,<2.0"

# OpenVoice deps — skip faster-whisper (av Cython incompatibility on Python 3.11)
pip install librosa==0.9.1 pydub==0.25.1 wavmark==0.0.3
pip install eng_to_ipa==0.0.2 langid==1.1.6

# Core ML and audio stack
pip install torch torchaudio soundfile

# Install OpenVoice package (editable, without pulling broken deps)
pip install -e OpenVoice/ --no-deps

# Install RTCC dependencies
pip install -r requirements.txt
```

### Step 4 — Download checkpoints

The S3 bucket in the original OpenVoice README is disabled. Download from Hugging Face instead:

```bash
pip install huggingface_hub

python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='myshell-ai/OpenVoice',
    repo_type='model',
    local_dir='./checkpoints_hf'
)
"

# Move to correct location
mkdir -p OpenVoice/openvoice/checkpoints
cp -r checkpoints_hf/checkpoints/* OpenVoice/openvoice/checkpoints/
```

After downloading, your checkpoint structure should look like:

```
OpenVoice/openvoice/checkpoints/
├── base_speakers/
│   └── EN/
│       ├── config.json
│       └── checkpoint.pth
└── converter/
    ├── config.json
    └── checkpoint.pth
```

### Step 5 — Verify installation

```bash
python -c "from openvoice.api import BaseSpeakerTTS, ToneColorConverter; print('OK')"
```

Expected output: `OK`

---

## 4. How to Use

### Text → Cloned Voice

Synthesise speech from text using the reference speaker's voice:

```bash
python rtcc_cli.py \
    --reference_audio audio/speaker.wav \
    --text "Hello, this is my cloned voice." \
    --output outputs/result.wav
```

### Audio → Cloned Voice (Voice Conversion)

Take the linguistic content of one audio file and render it in the reference speaker's voice:

```bash
python rtcc_cli.py \
    --reference_audio audio/speaker.wav \
    --input_audio audio/source_speech.wav \
    --output outputs/converted.wav
```

### Advanced Options

```bash
python rtcc_cli.py \
    --reference_audio audio/speaker.wav \
    --text "Hello world" \
    --output outputs/result.wav \
    --speaker cheerful \
    --speed 0.9 \
    --language English
```

Available `--speaker` styles: `default`, `whispering`, `shouting`, `excited`, `cheerful`, `terrified`, `angry`, `sad`, `friendly`

### Play Output (macOS)

```bash
afplay outputs/result.wav
```

---

## 5. CLI Reference

```
usage: rtcc_cli [-h]
                --reference_audio PATH
                --output PATH
                (--text TEXT | --input_audio PATH)
                [--openvoice_dir PATH]
                [--tts_config PATH]  [--tts_ckpt PATH]
                [--conv_config PATH] [--conv_ckpt PATH]
                [--speaker KEY] [--speed FLOAT] [--language LANG]
```

| Argument | Required | Description |
|----------|----------|-------------|
| `--reference_audio PATH` | ✅ | Reference `.wav` file whose voice will be cloned |
| `--output PATH` | ✅ | Output `.wav` file path |
| `--text TEXT` | ✅ or `--input_audio` | Text to synthesise in the cloned voice |
| `--input_audio PATH` | ✅ or `--text` | Source `.wav` to voice-convert into the cloned voice |
| `--openvoice_dir PATH` | ❌ | Root of OpenVoice repo (default: `./OpenVoice`) |
| `--tts_config PATH` | ❌ | Override `BaseSpeakerTTS` config.json path |
| `--tts_ckpt PATH` | ❌ | Override `BaseSpeakerTTS` checkpoint.pth path |
| `--conv_config PATH` | ❌ | Override `ToneColorConverter` config.json path |
| `--conv_ckpt PATH` | ❌ | Override `ToneColorConverter` checkpoint.pth path |
| `--speaker KEY` | ❌ | Base speaker style key (default: `default`) |
| `--speed FLOAT` | ❌ | Speaking rate multiplier, e.g. `0.8` = slower (default: `1.0`) |
| `--language LANG` | ❌ | Language tag passed to TTS (default: `English`) |

---

## 6. Troubleshooting

| Error / Symptom | Fix |
|-----------------|-----|
| `numpy` build fails on Python 3.14 | Use Python 3.11 — `numpy==1.22` cannot build on 3.12+ |
| `av` Cython compile error | Skip `faster-whisper`: install deps individually (see §3 Step 3) |
| `ModuleNotFoundError: openvoice` | Run `pip install -e OpenVoice/ --no-deps` |
| `FileNotFoundError: checkpoint.pth` | Download checkpoints from Hugging Face (see §3 Step 4) |
| `checkpoints not found` error pointing to wrong path | Your checkpoints are in `OpenVoice/openvoice/checkpoints/`, not `OpenVoice/checkpoints/` |
| `TypeError: argument of type 'torch.device' is not iterable` | Pass device as a plain string `"cpu"` or `"cuda"`, not `torch.device(...)` |
| `TypeError: extract_se() got unexpected keyword 'vad'` | Your OpenVoice version has no `vad` param — remove it from the call |
| `ValueError: not enough values to unpack` | `extract_se()` returns a single value — don't unpack as a tuple |
| `extract_se()` receives wrong type | Pass a **list** of paths: `extract_se([path])`, not `extract_se(path)` |
| Crackling or low quality output | Use a clean 3–10 second mono WAV as reference audio |
| Very slow inference | Expected on CPU: 30–120 s per utterance. Use a GPU for faster results |
| `FutureWarning: weight_norm deprecated` | Harmless — just a PyTorch deprecation notice, not an error |
| HF Hub unauthenticated warning | Harmless — set `HF_TOKEN` env var to suppress |

---

## 7. Hardware Requirements

| Hardware | Performance |
|----------|-------------|
| CPU (any modern x86 / Apple Silicon) | ~30–120 seconds per utterance — works, just slow |
| NVIDIA GPU (CUDA) | ~2–5 seconds per utterance — recommended |
| Apple Silicon (MPS) | CPU mode used automatically |
| RAM | Minimum 4 GB, 8 GB recommended |
| Disk | ~2 GB for model checkpoints |

No multi-GPU or distributed setup required.

---

## 8. Project Structure

```
RTCC/
├── audio/                        # Reference & input audio files
│   └── speaker.wav
├── outputs/                      # Generated .wav outputs
├── OpenVoice/                    # Cloned OpenVoice repo
│   └── openvoice/
│       └── checkpoints/
│           ├── base_speakers/
│           │   └── EN/
│           │       ├── config.json
│           │       └── checkpoint.pth
│           └── converter/
│               ├── config.json
│               └── checkpoint.pth
├── rtcc_cli.py                   # CLI entry point
├── voice_cloner.py               # Core cloning logic (VoiceCloner class)
├── openvoice_loader.py           # Model loading helpers
├── audio_utils.py                # Audio I/O utilities
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

---

## 9. Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | ≥2.0.0 | Core ML framework |
| `torchaudio` | ≥2.0.0 | Audio resampling |
| `soundfile` | ≥0.12.1 | WAV file I/O |
| `librosa` | ==0.9.1 | Audio analysis & resampling |
| `numpy` | ≥1.23, <2.0 | Numerical arrays |
| `wavmark` | ==0.0.3 | Audio watermarking (OpenVoice dep) |
| `pydub` | ==0.25.1 | Audio manipulation (OpenVoice dep) |
| `eng_to_ipa` | ==0.0.2 | English phoneme conversion (OpenVoice dep) |
| `langid` | ==1.1.6 | Language detection (OpenVoice dep) |
| `huggingface_hub` | latest | Checkpoint download |

---

## License

RTCC is a thin wrapper around [OpenVoice](https://github.com/myshell-ai/OpenVoice).  
See the [OpenVoice LICENSE](https://github.com/myshell-ai/OpenVoice/blob/main/LICENSE) for model usage terms.  
Both OpenVoice V1 and V2 are released under the **MIT License** as of April 2024.

---

---

## 10. Phase 1 — Usable Engine

> Upgrade from basic CLI prototype to a developer-friendly, low-latency local voice engine.  
> All changes are local and CLI-based — no APIs, no servers, no microservices.

### What Changed

| Area | Before | After |
|------|--------|-------|
| Output strategy | Generate full audio → save | Split → generate chunks → play progressively |
| Playback | Manual (`afplay`) after generation | `--play` flag streams audio during generation |
| Model loading | Reloaded on every run | Loaded once, cached in `VoiceCloner` |
| Speaker embedding | Re-extracted even for same reference | Cached; skipped if path unchanged |
| Warmup | None | `--warmup` flag runs dummy inference to pre-heat JIT |
| Mode selection | Auto-detected only | `--mode tts` / `--mode vc` (explicit or auto-detected) |
| Chunk control | Not available | `--chunk_size N` controls words per synthesis chunk |
| Temp files | Single shared tempfile | Shared tempdir, cleaned up on exit, minimal I/O |
| Logging | Basic print statements | `[RTCC]` prefixed progress logs throughout |

---

### 10.1 Install New Dependency

Phase 1 adds `sounddevice` for real-time audio playback:

```bash
pip install sounddevice
```

If `sounddevice` is not installed and `--play` is passed, RTCC will warn and continue without playback — it will not crash.

---

### 10.2 New CLI Flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--mode` | `tts` \| `vc` | auto-detected | Explicit pipeline mode. `tts` = text input, `vc` = audio input |
| `--play` | flag | `False` | Stream audio to speakers during generation via `sounddevice` |
| `--chunk_size` | int | `15` | Max words per TTS synthesis chunk. Set `0` to disable chunking |
| `--warmup` | flag | `False` | Run a dummy inference pass after loading to pre-heat model caches |

All existing flags (`--speaker`, `--speed`, `--language`, `--reference_audio`, `--text`, `--input_audio`, `--output`, model path overrides) are **unchanged and fully backward-compatible**.

---

### 10.3 Pseudo-Streaming (TTS Mode)

Previously, the full utterance was synthesised before anything was saved or played.

**Phase 1 pipeline:**

```
Text → split into chunks → for each chunk:
    TTS (base speech) → extract SE → tone convert → queue for playback
                                                          ↓
                                              play chunk immediately
                                              while next chunk generates
```

**Concurrency model:**

```
Main thread:   [gen chunk 1] [gen chunk 2] [gen chunk 3] …
Playback thread:        [play chunk 1] [play chunk 2] …
```

A `queue.Queue(maxsize=4)` sits between the two threads. The playback thread (`_PlaybackThread`) is a daemon that drains the queue via `sounddevice.play(..., blocking=True)`. Backpressure is applied automatically — the main thread blocks on `queue.put()` if the playback thread falls behind by more than 4 chunks.

**Time to first audio (approximate):**

| Hardware | Before (full utterance) | After (first chunk, 15 words) |
|----------|------------------------|-------------------------------|
| CPU | 30–120 s | 6–24 s |
| GPU (CUDA) | 2–5 s | 0.4–1 s |

---

### 10.4 Chunk Size Tuning

`--chunk_size` controls the trade-off between latency and overhead:

| `--chunk_size` | Best for |
|----------------|----------|
| `8–10` | Lowest time-to-first-audio; more TTS/converter call overhead |
| `15` (default) | Balanced — natural sentence breaks, low overhead |
| `20–25` | GPU users where per-chunk inference is already fast |
| `0` | Disable chunking entirely — identical to original behaviour |

Chunks are split first at sentence boundaries (`.` `!` `?`), then by word count if a sentence exceeds `chunk_size`.

---

### 10.5 Model Warmup

The first inference call is always slower than subsequent ones because PyTorch compiles CUDA graphs and allocates memory pools on first use. `--warmup` runs a silent dummy "Hello." through the full TTS → extract_se → convert pipeline immediately after model load, so the real request hits pre-warmed caches.

```bash
python rtcc_cli.py \
    --reference_audio audio/speaker.wav \
    --text "Hello world" \
    --output outputs/result.wav \
    --warmup \
    --play
```

Warmup adds ~5–30 s of startup cost but can reduce first-chunk latency by 20–50% on GPU.

---

### 10.6 Updated Example Usage

**TTS — pseudo-streaming with playback:**

```bash
python rtcc_cli.py \
    --mode tts \
    --reference_audio audio/obama.wav \
    --text "Hello world. I am here to celebrate this moment with all of you." \
    --output outputs/result.wav \
    --play \
    --chunk_size 10
```

**TTS — existing command, fully compatible (no new flags needed):**

```bash
python rtcc_cli.py \
    --reference_audio audio/obama.wav \
    --text "Hello world. I am here to celebrate" \
    --output outputs/result.wav \
    --speaker cheerful \
    --speed 0.9 \
    --language English
```

**TTS — with warmup for lower first-chunk latency:**

```bash
python rtcc_cli.py \
    --mode tts \
    --reference_audio audio/speaker.wav \
    --text "Hello world" \
    --output outputs/result.wav \
    --warmup \
    --play
```

**VC — voice conversion with immediate playback:**

```bash
python rtcc_cli.py \
    --mode vc \
    --reference_audio audio/speaker.wav \
    --input_audio audio/source.wav \
    --output outputs/converted.wav \
    --play
```

---

### 10.7 Updated Project Structure

```
RTCC/
├── audio/
│   └── speaker.wav
├── outputs/
├── OpenVoice/
│   └── openvoice/
│       └── checkpoints/
│           ├── base_speakers/EN/
│           └── converter/
├── rtcc_cli.py          # Updated — new flags: --mode, --play, --chunk_size, --warmup
├── voice_cloner.py      # Updated — streaming, caching, warmup, threaded playback
├── openvoice_loader.py  # Unchanged
├── audio_utils.py       # Unchanged
├── requirements.txt     # Updated — added sounddevice
└── README.md
```

---

### 10.8 Phase 1 Troubleshooting

| Error / Symptom | Fix |
|-----------------|-----|
| `ModuleNotFoundError: sounddevice` | Run `pip install sounddevice` |
| `--play` flag has no effect | `sounddevice` not installed — RTCC warns and skips playback gracefully |
| Audio plays out of order | Should not happen — playback queue is strictly ordered (FIFO) |
| Choppy playback between chunks | Increase `--chunk_size` to reduce inter-chunk gaps |
| `PortAudioError` on Linux | Install PortAudio: `sudo apt-get install libportaudio2` |
| `PortAudioError` on macOS | Usually self-resolving; try `brew install portaudio` if it persists |
| Warmup takes very long on CPU | Normal — skip `--warmup` on CPU; benefit is mainly for GPU |
| Temp files left on disk after crash | Stored in system tmpdir (e.g. `/tmp/rtcc_*/`) — safe to delete manually |