# RTCC — Real-Time Collaborative Cloner

> A minimal, fully local command-line tool for zero-shot voice cloning.  
> Powered by [OpenVoice](https://github.com/myshell-ai/OpenVoice) by MyShell AI & MIT.

---

## What Is RTCC?

RTCC takes a short reference audio sample of any speaker and can:

- **Generate speech from text** in the cloned voice *(Text → Voice)*
- **Convert any audio recording** into the cloned voice *(Audio → Voice Conversion)*

Built on **OpenVoice V2** — no server, no web UI, no API. Everything runs locally.

| Feature | Detail |
|---------|--------|
| Zero-shot cloning | No fine-tuning — just a 3–10 second reference clip |
| Fully offline | Runs on CPU or a single GPU |
| Two modes | Text synthesis and audio voice conversion |
| Language | English out of the box |
| Output | Standard `.wav` file |

---

## Quick Start

```bash
# Text → Cloned Voice
python rtcc_cli.py \
    --reference_audio audio/speaker.wav \
    --text "Hello, this is my cloned voice." \
    --output outputs/result.wav

# Audio → Cloned Voice (Voice Conversion)
python rtcc_cli.py \
    --reference_audio audio/speaker.wav \
    --input_audio audio/source_speech.wav \
    --output outputs/converted.wav
```

---

## Modes

### Text to Voice (TTS)
Synthesise speech from text using the reference speaker's voice.

### Audio to Voice Conversion (VC)
Take the linguistic content of one audio file and render it in the reference speaker's voice.

### Phase 1 — Streaming & Playback
Add `--play` to stream audio to your speakers during generation, and `--chunk_size` to tune latency.

```bash
python rtcc_cli.py \
    --mode tts \
    --reference_audio audio/speaker.wav \
    --text "Hello world. This streams as it generates." \
    --output outputs/result.wav \
    --play \
    --chunk_size 10
```

### Optional UI (Gradio)
A minimal browser interface is available as a developer convenience:

```bash
pip install gradio
python gradio_app.py
# Open http://localhost:7860
```

---

## Hardware

| Hardware | Performance |
|----------|-------------|
| CPU | ~30–120 s per utterance |
| NVIDIA GPU (CUDA) | ~2–5 s per utterance |
| Apple Silicon (MPS) | CPU mode used automatically |
| RAM | 4 GB minimum, 8 GB recommended |
| Disk | ~2 GB for checkpoints |

---

## Project Structure

```
RTCC/
├── audio/                  # Reference & input audio files
├── outputs/                # Generated .wav outputs
├── OpenVoice/              # Cloned OpenVoice repo + checkpoints
├── rtcc_cli.py             # CLI entry point
├── voice_cloner.py         # Core cloning logic
├── openvoice_loader.py     # Model loading helpers
├── audio_utils.py          # Audio I/O utilities
├── gradio_app.py           # Optional browser UI
├── requirements.txt
└── README.md
```

---

## Documentation

📖 **[Full Installation Guide & Reference →](docs/installation.md)**

Covers prerequisites, step-by-step setup, checkpoint download, CLI reference, all flags, troubleshooting, and Phase 1 details.

---

## License

RTCC wraps [OpenVoice](https://github.com/myshell-ai/OpenVoice), released under the **MIT License** as of April 2024.
