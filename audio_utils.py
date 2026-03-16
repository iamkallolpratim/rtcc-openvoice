"""
audio_utils.py
--------------
Utility functions for:
  - Loading audio files into numpy arrays / torch tensors
  - Resampling audio to a target sample rate
  - Saving audio tensors / numpy arrays to .wav files

Uses torchaudio as primary backend; falls back to soundfile + librosa
for resampling when needed.
"""

import numpy as np
import torch
import torchaudio
import soundfile as sf
import librosa


# Default sample rate expected by OpenVoice models
OPENVOICE_SR = 22050


def load_audio(path: str, target_sr: int = OPENVOICE_SR) -> tuple[np.ndarray, int]:
    """
    Load an audio file and return a mono float32 numpy array resampled to target_sr.

    Parameters
    ----------
    path      : path to the audio file
    target_sr : desired sample rate (default 22050)

    Returns
    -------
    (waveform_np, sample_rate)
      waveform_np : shape (N,), float32, values in [-1, 1]
      sample_rate : always equals target_sr
    """
    waveform, sr = torchaudio.load(path)          # (C, T)

    # Mix down to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)  # (1, T)

    # Resample if needed
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)

    waveform_np = waveform.squeeze(0).numpy().astype(np.float32)
    return waveform_np, target_sr


def load_audio_tensor(path: str, target_sr: int = OPENVOICE_SR) -> tuple[torch.Tensor, int]:
    """
    Same as load_audio but returns a torch.Tensor of shape (1, T).
    """
    waveform_np, sr = load_audio(path, target_sr)
    return torch.from_numpy(waveform_np).unsqueeze(0), sr


def save_audio(path: str, waveform, sample_rate: int = OPENVOICE_SR) -> None:
    """
    Save audio to a .wav file.

    Parameters
    ----------
    path        : output file path (should end in .wav)
    waveform    : numpy array (N,) or (1,N), OR torch.Tensor (1,T) / (T,)
    sample_rate : sample rate of the waveform
    """
    # Normalise to numpy float32 1-D
    if isinstance(waveform, torch.Tensor):
        waveform = waveform.detach().cpu().numpy()

    waveform = np.squeeze(waveform).astype(np.float32)

    # Clip to avoid clipping artefacts
    waveform = np.clip(waveform, -1.0, 1.0)

    sf.write(path, waveform, sample_rate, subtype="PCM_16")
    print(f"[audio_utils] Saved audio → {path}  (sr={sample_rate}, samples={len(waveform)})")


def resample_audio(waveform: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """
    Resample a 1-D numpy waveform using librosa.

    Parameters
    ----------
    waveform  : float32 numpy array
    orig_sr   : original sample rate
    target_sr : target sample rate

    Returns
    -------
    resampled numpy array
    """
    if orig_sr == target_sr:
        return waveform
    return librosa.resample(waveform, orig_sr=orig_sr, target_sr=target_sr)


def audio_numpy_to_tensor(waveform: np.ndarray) -> torch.Tensor:
    """Convert a (N,) numpy array to a (1, N) float32 torch tensor."""
    return torch.from_numpy(waveform.astype(np.float32)).unsqueeze(0)


def audio_tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert a (1, N) or (N,) torch tensor to a (N,) float32 numpy array."""
    return tensor.detach().cpu().squeeze().numpy().astype(np.float32)
