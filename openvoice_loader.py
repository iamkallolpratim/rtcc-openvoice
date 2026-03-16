"""
openvoice_loader.py
-------------------
Responsible for loading all required OpenVoice components:
  - BaseSpeakerTTS     : generates base speech from text
  - ToneColorConverter : transfers voice tone/timbre to target speaker
  - SE extractor       : extracts speaker embeddings from reference audio

NOTE: OpenVoice's API expects device as a plain string ("cpu" / "cuda"),
      NOT a torch.device object. All functions here use plain strings.
"""

import os
import sys
import torch


def get_device() -> str:
    """Return 'cuda' if a GPU is available, else 'cpu' (plain string)."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def ensure_openvoice_on_path(openvoice_dir: str = "./OpenVoice") -> None:
    """
    Add the OpenVoice repo root to sys.path so `from openvoice.api import ...`
    works correctly.

    Raises RuntimeError if the directory does not exist.
    """
    abs_dir = os.path.abspath(openvoice_dir)
    if not os.path.isdir(abs_dir):
        raise RuntimeError(
            f"OpenVoice directory not found at '{abs_dir}'.\n"
            "Please clone it first:\n"
            "  git clone https://github.com/myshell-ai/OpenVoice\n"
            "  cd OpenVoice && pip install -e ."
        )

    # Add the repo root (contains setup.py / pyproject.toml)
    if abs_dir not in sys.path:
        sys.path.insert(0, abs_dir)

    # Also add the inner openvoice/ subdirectory as a fallback so bare
    # `import openvoice` works even without an editable install.
    inner_dir = os.path.join(abs_dir, "openvoice")
    if os.path.isdir(inner_dir) and inner_dir not in sys.path:
        sys.path.insert(0, inner_dir)


def load_base_tts(config_path: str, checkpoint_path: str, device: str):
    """
    Load the BaseSpeakerTTS model.

    Parameters
    ----------
    config_path      : path to the TTS model config.json
    checkpoint_path  : path to the TTS model checkpoint.pth
    device           : plain string — "cpu" or "cuda"

    Returns
    -------
    BaseSpeakerTTS instance
    """
    from OpenVoice.openvoice.api import BaseSpeakerTTS  # noqa: E402

    tts_model = BaseSpeakerTTS(config_path, device=device)
    tts_model.load_ckpt(checkpoint_path)
    return tts_model


def load_tone_converter(config_path: str, checkpoint_path: str, device: str):
    """
    Load the ToneColorConverter model.

    Parameters
    ----------
    config_path      : path to the converter config.json
    checkpoint_path  : path to the converter checkpoint.pth
    device           : plain string — "cpu" or "cuda"

    Returns
    -------
    ToneColorConverter instance
    """
    from OpenVoice.openvoice.api import ToneColorConverter  # noqa: E402

    converter = ToneColorConverter(config_path, device=device)
    converter.load_ckpt(checkpoint_path)
    return converter


def load_se_extractor(device: str):
    """
    SE extraction is handled by ToneColorConverter.extract_se() directly.
    This stub exists for API symmetry; voice_cloner calls extract_se() itself.
    """
    return None  # SE extraction is handled by ToneColorConverter.extract_se()