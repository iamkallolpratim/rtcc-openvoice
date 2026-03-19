# RTCC — Installation Guide & Reference

> Full technical documentation for RTCC.  
> ← Back to [README](../README.md)

---

## Table of Contents

1. [Installation](#1-installation)
2. [Architecture](#2-architecture)
3. [CLI Reference](#3-cli-reference)
4. [Phase 1 — Streaming Engine](#4-phase-1--streaming-engine)
5. [Troubleshooting](#5-troubleshooting)
6. [Dependencies](#6-dependencies)

---

## 1. Installation

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

### Phase 1 — Install streaming dependency

Phase 1 adds `sounddevice` for real-time audio playback:

```bash
pip install sounddevice
```

If `sounddevice` is not installed and `--play` is passed, RTCC will warn and continue without playback — it will not crash.

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

OpenVoice encodes vocal timbre into a compact vector called a **Speaker Embedding (SE)**. The `ToneColorConverter` blends the *content* of one audio stream with the *timbre vector* of the reference speaker — producing speech that sounds like the reference speaker, regardless of what they are saying.

---

## 3. CLI Reference

```
usage: rtcc_cli [-h]
                --reference_audio PATH
                --output PATH
                (--text TEXT | --input_audio PATH)
                [--openvoice_dir PATH]
                [--tts_config PATH]  [--tts_ckpt PATH]
                [--conv_config PATH] [--conv_ckpt PATH]
                [--speaker KEY] [--speed FLOAT] [--language LANG]
                [--mode tts|vc] [--play] [--chunk_size N] [--warmup]
```

### Core Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `--reference_audio PATH` | ✅ | Reference `.wav` file whose voice will be cloned |
| `--output PATH` | ✅ | Output `.wav` file path |
| `--text TEXT` | ✅ or `--input_audio` | Text to synthesise in the cloned voice |
| `--input_audio PATH` | ✅ or `--text` | Source `.wav` to voice-convert into the cloned voice |

### Model Path Overrides

| Argument | Description |
|----------|-------------|
| `--openvoice_dir PATH` | Root of OpenVoice repo (default: `./OpenVoice`) |
| `--tts_config PATH` | Override `BaseSpeakerTTS` config.json path |
| `--tts_ckpt PATH` | Override `BaseSpeakerTTS` checkpoint.pth path |
| `--conv_config PATH` | Override `ToneColorConverter` config.json path |
| `--conv_ckpt PATH` | Override `ToneColorConverter` checkpoint.pth path |

### Voice Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--speaker KEY` | `default` | Base speaker style key |
| `--speed FLOAT` | `1.0` | Speaking rate multiplier, e.g. `0.8` = slower |
| `--language LANG` | `English` | Language tag passed to TTS |

Available `--speaker` styles: `default`, `whispering`, `shouting`, `excited`, `cheerful`, `terrified`, `angry`, `sad`, `friendly`

### Phase 1 Flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--mode` | `tts` \| `vc` | auto-detected | Explicit pipeline mode |
| `--play` | flag | `False` | Stream audio to speakers during generation |
| `--chunk_size` | int | `15` | Max words per TTS chunk. `0` disables chunking |
| `--warmup` | flag | `False` | Run a dummy inference pass after loading to pre-heat model caches |

### Example Commands

**Basic TTS:**
```bash
python rtcc_cli.py \
    --reference_audio audio/speaker.wav \
    --text "Hello, this is my cloned voice." \
    --output outputs/result.wav
```

**Voice Conversion:**
```bash
python rtcc_cli.py \
    --reference_audio audio/speaker.wav \
    --input_audio audio/source_speech.wav \
    --output outputs/converted.wav
```

**TTS with advanced options:**
```bash
python rtcc_cli.py \
    --reference_audio audio/speaker.wav \
    --text "Hello world" \
    --output outputs/result.wav \
    --speaker cheerful \
    --speed 0.9 \
    --language English
```

**Streaming TTS with playback:**
```bash
python rtcc_cli.py \
    --mode tts \
    --reference_audio audio/speaker.wav \
    --text "Hello world. This streams as it generates." \
    --output outputs/result.wav \
    --play \
    --chunk_size 10
```

**With warmup (recommended on GPU):**
```bash
python rtcc_cli.py \
    --mode tts \
    --reference_audio audio/speaker.wav \
    --text "Hello world" \
    --output outputs/result.wav \
    --warmup \
    --play
```

**Play output manually (macOS):**
```bash
afplay outputs/result.wav
```

---

## 4. Phase 1 — Streaming Engine

Phase 1 upgrades the basic CLI prototype to a developer-friendly, low-latency local voice engine. All changes are local and CLI-based — no APIs, no servers, no microservices.

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

### Pseudo-Streaming (TTS Mode)

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
Main thread:      [gen chunk 1] [gen chunk 2] [gen chunk 3] …
Playback thread:          [play chunk 1] [play chunk 2] …
```

A `queue.Queue(maxsize=4)` sits between the two threads. The playback thread (`_PlaybackThread`) is a daemon that drains the queue via `sounddevice.play(..., blocking=True)`. Backpressure is applied automatically — the main thread blocks on `queue.put()` if the playback thread falls behind by more than 4 chunks.

### Time to First Audio

| Hardware | Before (full utterance) | After (first chunk, 15 words) |
|----------|------------------------|-------------------------------|
| CPU | 30–120 s | 6–24 s |
| GPU (CUDA) | 2–5 s | 0.4–1 s |

### Chunk Size Tuning

`--chunk_size` controls the trade-off between latency and overhead:

| `--chunk_size` | Best for |
|----------------|----------|
| `8–10` | Lowest time-to-first-audio; more TTS/converter call overhead |
| `15` (default) | Balanced — natural sentence breaks, low overhead |
| `20–25` | GPU users where per-chunk inference is already fast |
| `0` | Disable chunking — identical to original behaviour |

Chunks split first at sentence boundaries (`.` `!` `?`), then by word count if a sentence exceeds `chunk_size`.

### Model Warmup

The first inference call is always slower because PyTorch compiles CUDA graphs and allocates memory pools on first use. `--warmup` runs a silent dummy "Hello." through the full TTS → extract_se → convert pipeline immediately after model load.

Warmup adds ~5–30 s of startup cost but can reduce first-chunk latency by 20–50% on GPU. Skip it on CPU — the benefit doesn't justify the cost.

---

## 5. Troubleshooting

### Setup & Installation

| Error / Symptom | Fix |
|-----------------|-----|
| `numpy` build fails | Use Python 3.11 — `numpy==1.22` cannot build on 3.12+ |
| `av` Cython compile error | Skip `faster-whisper`: install deps individually (see §1 Step 3) |
| `ModuleNotFoundError: openvoice` | Run `pip install -e OpenVoice/ --no-deps` |
| `FileNotFoundError: checkpoint.pth` | Download checkpoints from Hugging Face (see §1 Step 4) |
| `checkpoints not found` pointing to wrong path | Checkpoints live in `OpenVoice/openvoice/checkpoints/`, not `OpenVoice/checkpoints/` |

### Runtime Errors

| Error / Symptom | Fix |
|-----------------|-----|
| `TypeError: argument of type 'torch.device' is not iterable` | Pass device as a plain string `"cpu"` or `"cuda"`, not `torch.device(...)` |
| `TypeError: extract_se() got unexpected keyword 'vad'` | Your OpenVoice version has no `vad` param — remove it from the call |
| `ValueError: not enough values to unpack` | `extract_se()` returns a single value — don't unpack as a tuple |
| `extract_se()` receives wrong type | Pass a **list** of paths: `extract_se([path])`, not `extract_se(path)` |
| `FutureWarning: weight_norm deprecated` | Harmless — a PyTorch deprecation notice, not an error |
| HF Hub unauthenticated warning | Harmless — set `HF_TOKEN` env var to suppress |

### Audio Quality

| Symptom | Fix |
|---------|-----|
| Crackling or low quality output | Use a clean 3–10 second mono WAV as reference audio |
| Very slow inference | Expected on CPU (30–120 s). Use a GPU for faster results |

### Phase 1 / Playback

| Error / Symptom | Fix |
|-----------------|-----|
| `ModuleNotFoundError: sounddevice` | Run `pip install sounddevice` |
| `--play` flag has no effect | `sounddevice` not installed — RTCC warns and skips gracefully |
| Audio plays out of order | Should not happen — playback queue is strictly ordered (FIFO) |
| Choppy playback between chunks | Increase `--chunk_size` to reduce inter-chunk gaps |
| `PortAudioError` on Linux | `sudo apt-get install libportaudio2` |
| `PortAudioError` on macOS | Try `brew install portaudio` if it persists |
| Warmup takes very long on CPU | Normal — skip `--warmup` on CPU; benefit is mainly for GPU |
| Temp files left on disk after crash | Stored in system tmpdir (e.g. `/tmp/rtcc_*/`) — safe to delete manually |

---

## 6. Dependencies

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
| `sounddevice` | latest | Real-time audio playback (Phase 1) |
| `gradio` | latest | Optional browser UI |

---

← Back to [README](../README.md)