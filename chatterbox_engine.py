"""
chatterbox_engine.py — Chatterbox Integration for RTCC
------------------------------------------------------
Wraps Chatterbox TTS (https://github.com/resemble-ai/chatterbox) for:

  * High-quality text-to-speech with voice cloning (TTS mode)
  * Voice conversion (VC mode)
  * Chatterbox-Turbo for low-latency TTS with paralinguistic tags

Chatterbox is a family of SoTA open-source TTS models by Resemble AI:
  - Chatterbox (500M)       — English TTS with CFG & exaggeration control
  - Chatterbox-Turbo (350M) — Low-latency TTS with paralinguistic tags
  - Chatterbox-VC           — Voice conversion

Public API:
  engine = ChatterboxEngine(variant="turbo", device=...)
  engine.set_reference(path)
  engine.clone_from_text(text, output_path, ...)
  engine.clone_from_audio(input_path, output_path)
"""

import os
import sys
import time

import numpy as np
import soundfile as sf

# ── Lazy imports ────────────────────────────────────────────────────────────

_torch = None
_torchaudio = None


def _ensure_torch():
    global _torch, _torchaudio
    if _torch is None:
        import torch
        import torchaudio
        _torch = torch
        _torchaudio = torchaudio


# ── Logging ─────────────────────────────────────────────────────────────────

def _log(msg: str) -> None:
    print(f"[RTCC][Chatterbox] {msg}", flush=True)


# ── Optional sounddevice ────────────────────────────────────────────────────

try:
    import sounddevice as sd
    _SD_AVAILABLE = True
except ImportError:
    _SD_AVAILABLE = False


class ChatterboxEngine:
    """
    RTCC engine backed by Chatterbox for high-quality TTS and VC.

    Parameters
    ----------
    variant : str
        'standard' (500M, full controls), 'turbo' (350M, faster),
        or 'vc' (voice conversion only).
    device : str
        'cuda', 'cpu', or 'auto'.
    """

    VARIANTS = ("standard", "turbo", "vc")

    def __init__(
        self,
        variant: str = "turbo",
        device: str = "auto",
    ):
        if variant not in self.VARIANTS:
            raise ValueError(
                f"variant must be one of {self.VARIANTS}, got '{variant}'")

        _ensure_torch()
        self.variant = variant

        if device == "auto":
            self.device = "cuda" if _torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self._reference_path = None
        self._model = None
        self._model_loaded = False

    # ──────────────────────────────────────────────────────────────────────
    # Model loading
    # ──────────────────────────────────────────────────────────────────────

    def load_models(self) -> None:
        """Load Chatterbox model (downloads from HuggingFace on first run)."""
        if self._model_loaded:
            _log(f"Chatterbox-{self.variant} already loaded ✓")
            return

        _log(f"Loading Chatterbox-{self.variant} → {self.device}")
        _log("(First run downloads model weights from HuggingFace — this may take a few minutes)")
        t0 = time.perf_counter()

        # Pre-flight check: torch/torchvision compatibility
        try:
            import torchvision.ops  # triggers nms registration
        except (RuntimeError, ImportError) as e:
            if "torchvision::nms" in str(e) or "does not exist" in str(e):
                _torch_ver = _torch.__version__.split("+")[0]
                raise RuntimeError(
                    f"\n[RTCC][Chatterbox] torch/torchvision version mismatch!\n"
                    f"  Your torch={_torch.__version__}, but torchvision is incompatible.\n\n"
                    f"  Fix with:\n"
                    f"    pip install --force-reinstall torch=={_torch_ver} torchvision torchaudio\n\n"
                    f"  Or install a known-good set:\n"
                    f"    pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1\n"
                ) from e

        if self.variant == "turbo":
            from chatterbox.tts_turbo import ChatterboxTurboTTS
            self._model = ChatterboxTurboTTS.from_pretrained(
                device=self.device)
            # Patch s3tokenizer to force float32 — prevents dtype mismatch on CPU
            self._patch_s3tokenizer_dtype()

        elif self.variant == "standard":
            from chatterbox.tts import ChatterboxTTS
            self._model = ChatterboxTTS.from_pretrained(device=self.device)
            self._patch_s3tokenizer_dtype()

        elif self.variant == "vc":
            from chatterbox.vc import ChatterboxVC
            self._model = ChatterboxVC.from_pretrained(device=self.device)

        self._model_loaded = True
        _log(f"Loaded in {time.perf_counter() - t0:.1f}s ✓")

    def _patch_s3tokenizer_dtype(self) -> None:
        """
        Comprehensive float32 compatibility patches for Chatterbox on CPU.

        Chatterbox never casts audio to float32 internally, causing crashes
        when WAV files are loaded as float64.  Three patch points needed:

        1. torchaudio.load — returns float64 for certain WAV encodings
        2. s3tokenizer.log_mel_spectrogram — torch.stft produces float64
           magnitudes on CPU, but mel filter bank is float32
        3. voice_encoder.embeds_from_wavs — passes float64 mels to LSTM
           which requires float32
        """
        # ── Patch 1: torchaudio.load → always float32 ───────────────────
        try:
            import torchaudio as _ta
            if not hasattr(_ta, '_rtcc_patched'):
                _orig_load = _ta.load

                def _float32_load(*args, **kwargs):
                    wav, sr = _orig_load(*args, **kwargs)
                    if wav.dtype != _torch.float32:
                        wav = wav.to(_torch.float32)
                    return wav, sr

                _ta.load = _float32_load
                _ta._rtcc_patched = True
        except Exception:
            pass

        # ── Patch 2: s3tokenizer mel spectrogram → cast magnitudes ───────
        #    Source uses: self.n_fft (=400), self.window (registered buffer)
        #    Hop length is S3_HOP = 160 (from s3tokenizer module)
        try:
            from chatterbox.models.s3tokenizer import s3tokenizer as s3mod
            _orig_mel = s3mod.S3Tokenizer.log_mel_spectrogram

            def _patched_mel(self_tok, audio):
                import torch
                # Force input audio to float32
                if hasattr(audio, 'dtype') and audio.dtype != torch.float32:
                    audio = audio.to(torch.float32)

                # Call original — but if it fails due to dtype, do it manually
                try:
                    return _orig_mel(self_tok, audio)
                except RuntimeError:
                    pass

                # Manual implementation with forced float32
                n_fft = self_tok.n_fft  # 400
                hop = 160  # S3_HOP
                window = self_tok.window.to(
                    dtype=torch.float32, device=audio.device)

                stft = torch.stft(
                    audio, n_fft, hop, window=window,
                    return_complex=True,
                )
                magnitudes = stft[..., :-1].abs() ** 2
                magnitudes = magnitudes.to(torch.float32)

                mel_spec = self_tok._mel_filters.to(
                    dtype=torch.float32, device=audio.device
                ) @ magnitudes
                log_spec = torch.clamp(mel_spec, min=1e-10).log10()
                log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
                log_spec = (log_spec + 4.0) / 4.0
                return log_spec

            s3mod.S3Tokenizer.log_mel_spectrogram = _patched_mel
        except Exception:
            pass

        # ── Patch 3: voice encoder → cast at forward() level ────────────
        #    The LSTM requires float32 input. Mels come from librosa as
        #    float64 numpy arrays → torch tensors.  Patching forward()
        #    catches ALL code paths (embeds_from_mels, inference, etc.)
        try:
            from chatterbox.models.voice_encoder import voice_encoder as ve_mod
            _orig_forward = ve_mod.VoiceEncoder.forward

            def _patched_forward(self_ve, mels, *args, **kwargs):
                if mels.dtype != _torch.float32:
                    mels = mels.to(_torch.float32)
                return _orig_forward(self_ve, mels, *args, **kwargs)

            ve_mod.VoiceEncoder.forward = _patched_forward
        except Exception:
            pass

        _log("Patched Chatterbox for float32 CPU compatibility ✓")

    # ──────────────────────────────────────────────────────────────────────
    # Reference speaker
    # ──────────────────────────────────────────────────────────────────────

    def set_reference(self, reference_audio_path: str) -> None:
        """
        Set the reference speaker audio for voice cloning.

        Ensures the audio is saved as float32 WAV to avoid dtype mismatches
        in Chatterbox's mel spectrogram computation.
        """
        if not os.path.isfile(reference_audio_path):
            raise FileNotFoundError(
                f"Reference audio not found: {reference_audio_path}")

        _ensure_torch()

        # Load and check dtype — Chatterbox crashes if audio is float64
        wav, sr = _torchaudio.load(reference_audio_path)
        if wav.dtype != _torch.float32:
            _log(f"Converting reference audio from {wav.dtype} → float32")
            wav = wav.to(_torch.float32)

            # Save a float32 copy to a temp location
            import tempfile
            tmp_dir = tempfile.mkdtemp(prefix="rtcc_cb_")
            self._reference_path = os.path.join(
                tmp_dir, os.path.basename(reference_audio_path)
            )
            _torchaudio.save(self._reference_path, wav, sr)
            _log(
                f"Reference speaker set (float32 copy): {self._reference_path}")
        else:
            self._reference_path = reference_audio_path
            _log(f"Reference speaker set: {reference_audio_path}")

    # ──────────────────────────────────────────────────────────────────────
    # Text → Cloned voice
    # ──────────────────────────────────────────────────────────────────────

    def clone_from_text(
        self,
        text: str,
        output_path: str,
        exaggeration: float = 0.5,
        cfg_weight: float = 0.5,
        play: bool = False,
    ) -> None:
        """
        Synthesise text in the reference speaker's voice.

        Parameters
        ----------
        text : str
            Text to synthesise.  Turbo variant supports paralinguistic tags:
            [laugh], [chuckle], [cough], etc.
        output_path : str
            Output WAV path.
        exaggeration : float
            Emotion exaggeration (0.0–1.0+). Standard variant only.
        cfg_weight : float
            Classifier-free guidance weight (0.0–1.0). Standard variant only.
        play : bool
            Play audio after generation.
        """
        if self.variant == "vc":
            raise RuntimeError(
                "VC variant does not support TTS. Use 'standard' or 'turbo'.")

        self.load_models()

        if self._reference_path is None:
            _log("No reference set — using default speaker voice")

        _log(
            f"Generating TTS ({self.variant}): {text[:60]}{'…' if len(text) > 60 else ''}")
        t0 = time.perf_counter()

        kwargs = {
            "audio_prompt_path": self._reference_path} if self._reference_path else {}

        if self.variant == "standard":
            kwargs["exaggeration"] = exaggeration
            kwargs["cfg_weight"] = cfg_weight

        wav = self._model.generate(text, **kwargs)

        # Save
        os.makedirs(os.path.dirname(
            os.path.abspath(output_path)), exist_ok=True)
        _torchaudio.save(output_path, wav, self._model.sr)

        _log(f"TTS done in {time.perf_counter() - t0:.1f}s → {output_path}")

        if play:
            self._play_audio(output_path)

    # ──────────────────────────────────────────────────────────────────────
    # Audio → Cloned voice (voice conversion)
    # ──────────────────────────────────────────────────────────────────────

    def clone_from_audio(
        self,
        input_audio_path: str,
        output_path: str,
        play: bool = False,
    ) -> None:
        """
        Convert input audio to the reference speaker's voice.

        If the engine was created with variant='vc', uses the dedicated VC model.
        Otherwise, falls back to TTS-based resynthesis.
        """
        if self._reference_path is None:
            raise RuntimeError(
                "Call set_reference() before clone_from_audio()")

        # Use dedicated VC model if available
        if self.variant == "vc" or self._try_vc_model():
            self._vc_convert(input_audio_path, output_path)
        else:
            _log("Chatterbox VC model not available — this variant supports TTS only.")
            _log("Hint: Use --cb_variant vc for voice conversion, or use OpenVoice/MeanVC.")
            raise RuntimeError("Voice conversion requires variant='vc'")

        if play:
            self._play_audio(output_path)

    def _try_vc_model(self) -> bool:
        """Check if ChatterboxVC is importable."""
        try:
            from chatterbox.vc import ChatterboxVC
            return True
        except ImportError:
            return False

    def _vc_convert(self, input_path: str, output_path: str) -> None:
        """Run Chatterbox VC pipeline."""
        _log(f"Voice conversion: {input_path} → {output_path}")
        t0 = time.perf_counter()

        # Load VC model if not already loaded
        if self.variant != "vc" or not self._model_loaded:
            from chatterbox.vc import ChatterboxVC
            vc_model = ChatterboxVC.from_pretrained(device=self.device)
        else:
            vc_model = self._model

        # Run VC — Chatterbox VC.generate() expects file path for audio, not tensor
        result = vc_model.generate(
            audio=input_path,
            target_voice_path=self._reference_path,
        )

        # Save
        os.makedirs(os.path.dirname(
            os.path.abspath(output_path)), exist_ok=True)
        if hasattr(vc_model, 'sr'):
            _torchaudio.save(output_path, result, vc_model.sr)
        else:
            _torchaudio.save(output_path, result, 22050)

        _log(f"VC done in {time.perf_counter() - t0:.1f}s → {output_path}")

    # ──────────────────────────────────────────────────────────────────────
    # Playback helper
    # ──────────────────────────────────────────────────────────────────────

    def _play_audio(self, path: str) -> None:
        if not _SD_AVAILABLE:
            _log("sounddevice not installed — skipping playback")
            return
        data, sr = sf.read(path, dtype="float32")
        _log("Playing audio …")
        sd.play(data, samplerate=sr, blocking=True)

    # ──────────────────────────────────────────────────────────────────────
    # Info
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def engine_name() -> str:
        return "chatterbox"

    def sample_rate(self) -> int:
        if self._model_loaded and hasattr(self._model, 'sr'):
            return self._model.sr
        return 22050

    @staticmethod
    def is_available() -> bool:
        """Check if chatterbox-tts is installed."""
        try:
            import chatterbox
            return True
        except ImportError:
            return False
