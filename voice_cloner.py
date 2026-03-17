"""
voice_cloner.py  (Phase 1 — Usable Engine)
------------------------------------------
Core voice-cloning logic. Wraps OpenVoice and adds:

  * Model caching  — models are loaded once; reused across calls
  * Warmup         — optional dummy inference to pre-heat JIT caches
  * Pseudo-streaming (TTS mode) — text split into sentence chunks;
    each chunk is generated + converted then queued for playback
    BEFORE the next chunk starts, so audio begins playing within
    ~one-chunk latency instead of after the full utterance.
  * Threaded playback — a dedicated daemon thread drains the queue
    with sounddevice (falls back to soundfile write-only when
    sounddevice is not installed or --play is not set).
  * Minimal file I/O — intermediate WAVs are written to a single
    shared tempdir that is cleaned up on exit.
  * Friendly RTCC-prefix log messages throughout.

Public API (unchanged from v0):
  cloner = VoiceCloner(...)
  cloner.set_reference(path)
  cloner.clone_from_text(text, output_path, ...)
  cloner.clone_from_audio(input_audio_path, output_path)

New parameters on clone_from_text:
  chunk_size : int   — max words per TTS chunk (default 15)
  play       : bool  — stream audio to speakers while generating

New methods:
  cloner.warmup()
"""

import os
import re
import sys
import queue
import tempfile
import threading
import time
import atexit

import numpy as np
import soundfile as sf

# ── Optional sounddevice (graceful fallback) ─────────────────────────────────
try:
    import sounddevice as sd
    _SD_AVAILABLE = True
except ImportError:
    _SD_AVAILABLE = False

# ── Default checkpoint paths ──────────────────────────────────────────────────
_DEFAULT_CKPT_BASE      = "OpenVoice/openvoice/checkpoints/base_speakers/EN"
_DEFAULT_CKPT_CONVERTER = "OpenVoice/openvoice/checkpoints/converter"

_DEFAULT_TTS_CONFIG  = os.path.join(_DEFAULT_CKPT_BASE,      "config.json")
_DEFAULT_TTS_CKPT    = os.path.join(_DEFAULT_CKPT_BASE,      "checkpoint.pth")
_DEFAULT_CONV_CONFIG = os.path.join(_DEFAULT_CKPT_CONVERTER, "config.json")
_DEFAULT_CONV_CKPT   = os.path.join(_DEFAULT_CKPT_CONVERTER, "checkpoint.pth")

# Sentinel to signal the playback thread that the queue is exhausted
_PLAYBACK_DONE = object()

# OpenVoice models work at 22 050 Hz
_MODEL_SR = 22050


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _log(msg: str) -> None:
    print(f"[RTCC] {msg}", flush=True)


def _split_into_chunks(text: str, chunk_size: int) -> list[str]:
    """
    Split text into chunks of at most `chunk_size` words.

    Strategy:
      1. First split on sentence boundaries (. ! ?) to keep natural pauses.
      2. If a sentence is longer than chunk_size words, sub-split by word count.

    Returns a list of non-empty stripped strings.
    """
    # Sentence split — keep the terminal punctuation attached
    raw_sentences = re.split(r'(?<=[.!?])\s+', text.strip())

    chunks: list[str] = []
    for sentence in raw_sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        words = sentence.split()
        if len(words) <= chunk_size:
            chunks.append(sentence)
        else:
            # Break long sentence into word-count sub-chunks
            for start in range(0, len(words), chunk_size):
                part = " ".join(words[start : start + chunk_size])
                chunks.append(part)

    return [c for c in chunks if c]


def _load_wav_numpy(path: str) -> tuple[np.ndarray, int]:
    """Read a WAV file → (float32 numpy array, sample_rate)."""
    data, sr = sf.read(path, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)          # stereo → mono
    return data, sr


def _concatenate_wavs(paths: list[str]) -> tuple[np.ndarray, int]:
    """Load and concatenate a list of WAV files in order."""
    arrays, sr = [], _MODEL_SR
    for p in paths:
        arr, sr = _load_wav_numpy(p)
        arrays.append(arr)
    return np.concatenate(arrays, axis=0), sr


# ─────────────────────────────────────────────────────────────────────────────
# Playback thread
# ─────────────────────────────────────────────────────────────────────────────

class _PlaybackThread(threading.Thread):
    """
    Daemon thread that drains an audio queue and plays each chunk
    through sounddevice.  Chunks are numpy float32 arrays.

    If sounddevice is not available the thread exits immediately —
    the caller is responsible for checking _SD_AVAILABLE beforehand.
    """

    def __init__(self, audio_queue: queue.Queue, samplerate: int = _MODEL_SR):
        super().__init__(daemon=True, name="rtcc-playback")
        self._q  = audio_queue
        self._sr = samplerate

    def run(self) -> None:
        if not _SD_AVAILABLE:
            return
        while True:
            item = self._q.get()
            if item is _PLAYBACK_DONE:
                self._q.task_done()
                break
            audio_arr = item
            try:
                _log("Playing audio chunk …")
                sd.play(audio_arr, samplerate=self._sr, blocking=True)
            except Exception as exc:  # noqa: BLE001
                print(f"[RTCC][warn] sounddevice playback error: {exc}", file=sys.stderr)
            finally:
                self._q.task_done()


# ─────────────────────────────────────────────────────────────────────────────
# VoiceCloner
# ─────────────────────────────────────────────────────────────────────────────

class VoiceCloner:
    """
    Minimal RTCC voice cloner — Phase 1 (Usable Engine).

    Models are loaded once in __init__ and reused for all subsequent calls.

    Parameters
    ----------
    tts_config_path   : path to BaseSpeakerTTS config.json
    tts_ckpt_path     : path to BaseSpeakerTTS checkpoint.pth
    conv_config_path  : path to ToneColorConverter config.json
    conv_ckpt_path    : path to ToneColorConverter checkpoint.pth
    device            : "cpu" or "cuda" (auto-detected if None)
    openvoice_dir     : root directory of the cloned OpenVoice repo
    """

    def __init__(
        self,
        tts_config_path:  str = _DEFAULT_TTS_CONFIG,
        tts_ckpt_path:    str = _DEFAULT_TTS_CKPT,
        conv_config_path: str = _DEFAULT_CONV_CONFIG,
        conv_ckpt_path:   str = _DEFAULT_CONV_CKPT,
        device:           str | None = None,
        openvoice_dir:    str = "./OpenVoice",
    ):
        from openvoice_loader import (
            ensure_openvoice_on_path,
            get_device,
            load_base_tts,
            load_tone_converter,
        )

        ensure_openvoice_on_path(openvoice_dir)

        self.device: str = device or get_device()
        _log(f"Using device: {self.device}")

        _log("Loading model… (BaseSpeakerTTS)")
        self.tts = load_base_tts(tts_config_path, tts_ckpt_path, self.device)

        _log("Loading model… (ToneColorConverter)")
        self.converter = load_tone_converter(conv_config_path, conv_ckpt_path, self.device)

        # Speaker embedding — set by set_reference()
        self.reference_se = None
        self._reference_path: str | None = None

        # Shared tempdir for intermediate files — cleaned up on exit
        self._tmpdir = tempfile.mkdtemp(prefix="rtcc_")
        atexit.register(self._cleanup_tmpdir)

        # Monotonically increasing counter for unique temp filenames
        self._tmp_counter = 0

        _log("Models loaded ✓")

    # ──────────────────────────────────────────────────────────────────────
    # Warmup
    # ──────────────────────────────────────────────────────────────────────

    def warmup(self) -> None:
        """
        Run a short dummy inference pass to pre-heat model JIT caches.

        Call this once after __init__ to reduce the latency of the first
        real request.  The warmup text is discarded immediately.
        """
        if self.reference_se is None:
            _log("Warmup skipped — no reference audio set yet.")
            return

        _log("Warming up models …")
        t0 = time.perf_counter()

        dummy_base = self._tmp_path("warmup_base.wav")
        dummy_out  = self._tmp_path("warmup_out.wav")
        try:
            self.tts.tts("Hello.", dummy_base, speaker="default",
                         language="English", speed=1.0)
            src_se = self.converter.extract_se([dummy_base])
            self.converter.convert(
                audio_src_path=dummy_base,
                src_se=src_se,
                tgt_se=self.reference_se,
                output_path=dummy_out,
                message="@MyShell",
            )
        finally:
            for p in (dummy_base, dummy_out):
                _safe_remove(p)

        _log(f"Warmup complete in {time.perf_counter() - t0:.1f}s ✓")

    # ──────────────────────────────────────────────────────────────────────
    # Reference speaker
    # ──────────────────────────────────────────────────────────────────────

    def set_reference(self, reference_audio_path: str) -> None:
        """
        Extract and cache the speaker embedding from reference audio.

        Re-uses the cached embedding if called with the same path twice —
        avoids redundant I/O on repeated calls.
        """
        if reference_audio_path == self._reference_path and self.reference_se is not None:
            _log("Speaker embedding already cached ✓")
            return

        _log(f"Extracting speaker embedding from: {reference_audio_path}")
        self.reference_se = self.converter.extract_se([reference_audio_path])
        self._reference_path = reference_audio_path
        _log("Speaker embedding extracted ✓")

    # ──────────────────────────────────────────────────────────────────────
    # Text → Cloned voice  (pseudo-streaming)
    # ──────────────────────────────────────────────────────────────────────

    def clone_from_text(
        self,
        text: str,
        output_path: str,
        speaker_key: str = "default",
        speed: float = 1.0,
        language: str = "English",
        chunk_size: int = 15,
        play: bool = False,
    ) -> None:
        """
        Generate speech from text and apply the reference speaker's voice.

        When `chunk_size` > 0 the text is split into sentence-level chunks.
        Each chunk is synthesised and tone-converted individually, then
        played immediately (if `play=True`) while the next chunk is being
        generated.  This makes the first audio audible after only one
        chunk's worth of inference, not after the full utterance.

        The chunks are concatenated and saved to `output_path` at the end.

        Parameters
        ----------
        text        : input text (any length)
        output_path : final merged WAV output
        speaker_key : OpenVoice speaker style (default, cheerful, …)
        speed       : speaking rate multiplier
        language    : language string for TTS
        chunk_size  : max words per synthesis chunk (0 = no chunking)
        play        : if True, stream audio via sounddevice
        """
        self._check_reference()

        if play and not _SD_AVAILABLE:
            print(
                "[RTCC][warn] --play requested but sounddevice is not installed.\n"
                "             Install it with:  pip install sounddevice\n"
                "             Continuing without real-time playback.",
                file=sys.stderr,
            )
            play = False

        # Split text into chunks
        chunks = _split_into_chunks(text, chunk_size) if chunk_size > 0 else [text]
        n = len(chunks)
        _log(f"Text split into {n} chunk(s) (chunk_size={chunk_size} words)")

        # Set up playback queue + thread
        audio_queue: queue.Queue = queue.Queue(maxsize=4)
        player: _PlaybackThread | None = None
        if play:
            player = _PlaybackThread(audio_queue, samplerate=_MODEL_SR)
            player.start()

        chunk_paths: list[str] = []

        try:
            for idx, chunk_text in enumerate(chunks, 1):
                _log(f'Generating chunk {idx}/{n}: "{chunk_text[:60]}"')

                base_path = self._tmp_path(f"base_{idx}.wav")
                out_path  = self._tmp_path(f"chunk_{idx}.wav")

                # 1. TTS — text → neutral base speech
                self.tts.tts(
                    chunk_text,
                    base_path,
                    speaker=speaker_key,
                    language=language,
                    speed=speed,
                )

                # 2. Extract source SE from the freshly synthesised base
                src_se = self.converter.extract_se([base_path])

                # 3. Tone conversion — apply reference speaker timbre
                self.converter.convert(
                    audio_src_path=base_path,
                    src_se=src_se,
                    tgt_se=self.reference_se,
                    output_path=out_path,
                    message="@MyShell",
                )

                # Discard base speech immediately — not needed anymore
                _safe_remove(base_path)

                chunk_paths.append(out_path)

                # 4. Queue for playback (non-blocking put with backpressure)
                if play:
                    audio_arr, _ = _load_wav_numpy(out_path)
                    audio_queue.put(audio_arr)   # blocks if queue is full (≤4)

            # Signal playback thread that generation is finished
            if play and player is not None:
                audio_queue.put(_PLAYBACK_DONE)
                player.join()

        except Exception:
            # Signal playback thread to stop on error
            if play and player is not None and player.is_alive():
                audio_queue.put(_PLAYBACK_DONE)
            raise

        # 5. Merge all chunks and save final output
        _log(f"Merging {len(chunk_paths)} chunk(s) → {output_path}")
        merged, sr = _concatenate_wavs(chunk_paths)
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        sf.write(output_path, merged, sr, subtype="PCM_16")

        # Clean up chunk temps
        for p in chunk_paths:
            _safe_remove(p)

        _log(f"Done → {output_path}")

    # ──────────────────────────────────────────────────────────────────────
    # Audio → Cloned voice (voice conversion — no chunking needed)
    # ──────────────────────────────────────────────────────────────────────

    def clone_from_audio(
        self,
        input_audio_path: str,
        output_path: str,
        play: bool = False,
    ) -> None:
        """
        Convert the voice in input_audio_path to the reference speaker's voice.

        Pipeline: input.wav → extract src SE → ToneColorConverter → output.wav
        """
        self._check_reference()

        _log(f"Extracting source embedding from: {input_audio_path}")
        src_se = self.converter.extract_se([input_audio_path])

        _log("Applying tone-color conversion …")
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        self.converter.convert(
            audio_src_path=input_audio_path,
            src_se=src_se,
            tgt_se=self.reference_se,
            output_path=output_path,
            message="@MyShell",
        )
        _log(f"Done → {output_path}")

        if play:
            if not _SD_AVAILABLE:
                print(
                    "[RTCC][warn] --play requested but sounddevice is not installed.",
                    file=sys.stderr,
                )
            else:
                _log("Playing audio …")
                audio_arr, sr = _load_wav_numpy(output_path)
                sd.play(audio_arr, samplerate=sr, blocking=True)

    # ──────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────

    def _check_reference(self) -> None:
        if self.reference_se is None:
            raise RuntimeError(
                "No reference audio set. Call set_reference(path) first."
            )

    def _tmp_path(self, name: str) -> str:
        """Return a unique path inside the shared tempdir."""
        self._tmp_counter += 1
        return os.path.join(self._tmpdir, f"{self._tmp_counter:04d}_{name}")

    def _cleanup_tmpdir(self) -> None:
        import shutil
        try:
            shutil.rmtree(self._tmpdir, ignore_errors=True)
        except Exception:  # noqa: BLE001
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────────────────────

def _safe_remove(path: str) -> None:
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except OSError:
        pass