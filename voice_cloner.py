"""
voice_cloner.py
---------------
Core voice-cloning logic. Wraps OpenVoice models and exposes two methods:

  clone_from_text(text, output_path)
      text → BaseSpeakerTTS → ToneColorConverter(reference SE) → wav

  clone_from_audio(input_audio_path, output_path)
      audio → ToneColorConverter(reference SE) → wav

Call set_reference(path) before either clone method.
"""

import os
import tempfile

import torch

from audio_utils import OPENVOICE_SR

# ---------------------------------------------------------------------------
# Default checkpoint paths — relative to where you run rtcc_cli.py
# ---------------------------------------------------------------------------
_DEFAULT_CKPT_BASE      = "OpenVoice/openvoice/checkpoints/base_speakers/EN"
_DEFAULT_CKPT_CONVERTER = "OpenVoice/openvoice/checkpoints/converter"

_DEFAULT_TTS_CONFIG  = os.path.join(_DEFAULT_CKPT_BASE,      "config.json")
_DEFAULT_TTS_CKPT    = os.path.join(_DEFAULT_CKPT_BASE,      "checkpoint.pth")
_DEFAULT_CONV_CONFIG = os.path.join(_DEFAULT_CKPT_CONVERTER, "config.json")
_DEFAULT_CONV_CKPT   = os.path.join(_DEFAULT_CKPT_CONVERTER, "checkpoint.pth")


class VoiceCloner:
    """
    Minimal RTCC voice cloner backed by OpenVoice.

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

        # device is always a plain string — OpenVoice API requires this
        self.device: str = device or get_device()
        print(f"[VoiceCloner] Using device: {self.device}")

        print("[VoiceCloner] Loading BaseSpeakerTTS …")
        self.tts = load_base_tts(tts_config_path, tts_ckpt_path, self.device)

        print("[VoiceCloner] Loading ToneColorConverter …")
        self.converter = load_tone_converter(conv_config_path, conv_ckpt_path, self.device)

        # Set by set_reference()
        self.reference_se = None

    # ------------------------------------------------------------------
    # Reference audio
    # ------------------------------------------------------------------

    def set_reference(self, reference_audio_path: str) -> None:
        """Extract and store the speaker embedding from reference audio."""
        print(f"[VoiceCloner] Extracting speaker embedding from: {reference_audio_path}")
        # extract_se expects a LIST of paths, not a single string
        self.reference_se = self.converter.extract_se([reference_audio_path])
        print("[VoiceCloner] Speaker embedding extracted ✓")

    # ------------------------------------------------------------------
    # Text → Cloned voice
    # ------------------------------------------------------------------

    def clone_from_text(
        self,
        text: str,
        output_path: str,
        speaker_key: str = "default",
        speed: float = 1.0,
        language: str = "English",
    ) -> None:
        """
        Generate speech from text and apply the reference speaker's voice.

        Pipeline: text → BaseSpeakerTTS → ToneColorConverter → output.wav
        """
        self._check_reference()

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            print(f"[VoiceCloner] Synthesising base speech for: '{text[:60]}…'")
            self.tts.tts(
                text,
                tmp_path,
                speaker=speaker_key,
                language=language,
                speed=speed,
            )

            # extract_se expects a LIST of paths
            src_se = self.converter.extract_se([tmp_path])

            print("[VoiceCloner] Applying tone-color conversion …")
            self.converter.convert(
                audio_src_path=tmp_path,
                src_se=src_se,
                tgt_se=self.reference_se,
                output_path=output_path,
                message="@MyShell",
            )
            print(f"[VoiceCloner] Done → {output_path}")

        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    # ------------------------------------------------------------------
    # Audio → Cloned voice (voice conversion)
    # ------------------------------------------------------------------

    def clone_from_audio(
        self,
        input_audio_path: str,
        output_path: str,
    ) -> None:
        """
        Convert the voice in input_audio_path to the reference speaker's voice.

        Pipeline: input.wav → extract src SE → ToneColorConverter → output.wav
        """
        self._check_reference()

        print(f"[VoiceCloner] Extracting source embedding from: {input_audio_path}")
        # extract_se expects a LIST of paths
        src_se = self.converter.extract_se([input_audio_path])

        print("[VoiceCloner] Applying tone-color conversion …")
        self.converter.convert(
            audio_src_path=input_audio_path,
            src_se=src_se,
            tgt_se=self.reference_se,
            output_path=output_path,
            message="@MyShell",
        )
        print(f"[VoiceCloner] Done → {output_path}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_reference(self) -> None:
        if self.reference_se is None:
            raise RuntimeError(
                "No reference audio set. Call set_reference(path) first."
            )