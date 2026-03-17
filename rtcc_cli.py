#!/usr/bin/env python3
"""
rtcc_cli.py  (Phase 1 — Usable Engine)
---------------------------------------
Real-Time Collaborative Cloner — command-line interface.

TEXT → CLONED VOICE  (TTS mode, pseudo-streaming):
    python rtcc_cli.py \\
        --mode tts \\
        --reference_audio speaker.wav \\
        --text "Hello, this is my cloned voice." \\
        --output outputs/result.wav \\
        --play

AUDIO → CLONED VOICE  (voice-conversion mode):
    python rtcc_cli.py \\
        --mode vc \\
        --reference_audio speaker.wav \\
        --input_audio speech.wav \\
        --output outputs/converted.wav \\
        --play

LEGACY COMPAT — original positional args still work (mode auto-detected):
    python rtcc_cli.py \\
        --reference_audio speaker.wav \\
        --text "Hello" \\
        --output outputs/result.wav

What's new in Phase 1
---------------------
  --mode tts | vc      : explicit mode selection (auto-detected if omitted)
  --play               : start audio playback before generation finishes
  --chunk_size N       : words per synthesis chunk  (default: 15)
  --warmup             : run a dummy inference pass to pre-heat caches
  Friendly [RTCC] log  : progress visible throughout
  Model caching        : models loaded once, reused across pipeline steps
"""

import argparse
import os
import sys
import time


# ─────────────────────────────────────────────────────────────────────────────
# Logging helper
# ─────────────────────────────────────────────────────────────────────────────

def _log(msg: str) -> None:
    print(f"[RTCC] {msg}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="rtcc_cli",
        description="RTCC – Real-Time Collaborative Cloner  (Phase 1: Usable Engine)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── Mode ─────────────────────────────────────────────────────────────────
    p.add_argument(
        "--mode",
        choices=["tts", "vc"],
        default=None,
        help=(
            "Pipeline mode.  "
            "'tts' = text → cloned speech (pseudo-streaming).  "
            "'vc'  = audio → cloned speech (voice conversion).  "
            "Auto-detected from --text / --input_audio if omitted."
        ),
    )

    # ── Required ─────────────────────────────────────────────────────────────
    p.add_argument(
        "--reference_audio",
        required=True,
        metavar="PATH",
        help="Reference .wav file whose voice will be cloned.",
    )
    p.add_argument(
        "--output",
        required=True,
        metavar="PATH",
        help="Output .wav file path.",
    )

    # ── Input (text or audio — one required) ─────────────────────────────────
    p.add_argument(
        "--text",
        metavar="TEXT",
        default=None,
        help="[TTS mode] Text to synthesise in the cloned voice.",
    )
    p.add_argument(
        "--input_audio",
        metavar="PATH",
        default=None,
        help="[VC mode] Source audio whose content is converted to the cloned voice.",
    )

    # ── Streaming / playback ──────────────────────────────────────────────────
    p.add_argument(
        "--play",
        action="store_true",
        default=False,
        help=(
            "Stream audio to speakers during generation.  "
            "In TTS mode playback starts after the first chunk is ready, "
            "not after the full utterance.  Requires sounddevice."
        ),
    )
    p.add_argument(
        "--chunk_size",
        type=int,
        default=15,
        metavar="N",
        help=(
            "[TTS mode] Max words per synthesis chunk.  "
            "Smaller = lower first-audio latency but more overhead.  "
            "Default: 15.  Set 0 to disable chunking."
        ),
    )

    # ── Warmup ───────────────────────────────────────────────────────────────
    p.add_argument(
        "--warmup",
        action="store_true",
        default=False,
        help=(
            "Run a short dummy inference pass after loading models to pre-heat "
            "JIT caches.  Adds ~5–30 s startup cost but lowers first-chunk latency."
        ),
    )

    # ── TTS style ─────────────────────────────────────────────────────────────
    p.add_argument(
        "--speaker",
        default="default",
        metavar="KEY",
        help=(
            "Base speaker style.  "
            "Choices: default, whispering, shouting, excited, cheerful, "
            "terrified, angry, sad, friendly.  (default: default)"
        ),
    )
    p.add_argument(
        "--speed",
        type=float,
        default=1.0,
        metavar="FLOAT",
        help="Speaking rate multiplier, e.g. 0.9 = 10%% slower.  (default: 1.0)",
    )
    p.add_argument(
        "--language",
        default="English",
        metavar="LANG",
        help="TTS language string.  (default: English)",
    )

    # ── Model paths (optional overrides) ─────────────────────────────────────
    p.add_argument("--openvoice_dir", default="./OpenVoice", metavar="PATH",
                   help="Root of the cloned OpenVoice repo.  (default: ./OpenVoice)")
    p.add_argument("--tts_config",  default=None, metavar="PATH",
                   help="Override BaseSpeakerTTS config.json path.")
    p.add_argument("--tts_ckpt",    default=None, metavar="PATH",
                   help="Override BaseSpeakerTTS checkpoint.pth path.")
    p.add_argument("--conv_config", default=None, metavar="PATH",
                   help="Override ToneColorConverter config.json path.")
    p.add_argument("--conv_ckpt",   default=None, metavar="PATH",
                   help="Override ToneColorConverter checkpoint.pth path.")

    return p


# ─────────────────────────────────────────────────────────────────────────────
# Mode resolution  (--mode is optional for backward compat)
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_mode(args: argparse.Namespace) -> str:
    """
    Return the effective mode string ("tts" or "vc").

    Rules:
      1. If --mode is given explicitly, use it — but validate that the
         matching input arg is also present.
      2. If --mode is omitted, auto-detect from --text / --input_audio.
      3. If neither input arg is present, print a helpful error and exit.
    """
    if args.mode is not None:
        # Explicit mode — validate
        if args.mode == "tts" and args.text is None:
            _abort("--mode tts requires --text TEXT")
        if args.mode == "vc" and args.input_audio is None:
            _abort("--mode vc requires --input_audio PATH")
        return args.mode

    # Auto-detect
    if args.text and args.input_audio:
        _abort("Provide either --text or --input_audio, not both.")
    if args.text:
        return "tts"
    if args.input_audio:
        return "vc"

    _abort(
        "You must supply either:\n"
        "  --text TEXT          for TTS mode\n"
        "  --input_audio PATH   for voice-conversion mode\n"
        "(Or set --mode tts / --mode vc explicitly.)"
    )


def _abort(msg: str) -> None:
    print(f"\n[RTCC] ERROR — {msg}\n", file=sys.stderr)
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint path resolution
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_ckpt_paths(args: argparse.Namespace) -> dict[str, str]:
    base      = os.path.join(args.openvoice_dir, "openvoice", "checkpoints",
                             "base_speakers", "EN")
    converter = os.path.join(args.openvoice_dir, "openvoice", "checkpoints",
                             "converter")
    return {
        "tts_config_path":  args.tts_config  or os.path.join(base,      "config.json"),
        "tts_ckpt_path":    args.tts_ckpt    or os.path.join(base,      "checkpoint.pth"),
        "conv_config_path": args.conv_config or os.path.join(converter, "config.json"),
        "conv_ckpt_path":   args.conv_ckpt   or os.path.join(converter, "checkpoint.pth"),
    }


def _validate_paths(args: argparse.Namespace, ckpt: dict[str, str], mode: str) -> None:
    errors: list[str] = []

    if not os.path.isfile(args.reference_audio):
        errors.append(f"  reference_audio not found  : {args.reference_audio}")

    if mode == "vc" and not os.path.isfile(args.input_audio):
        errors.append(f"  input_audio not found      : {args.input_audio}")

    for label, path in ckpt.items():
        if not os.path.isfile(path):
            errors.append(f"  model file not found ({label}): {path}")

    if errors:
        print("\n[RTCC] ERROR – missing files:\n" + "\n".join(errors))
        print(
            "\nPlease ensure:\n"
            "  1. OpenVoice is cloned to the --openvoice_dir path.\n"
            "  2. Checkpoints are downloaded (see README.md).\n"
            "  3. All audio paths point to existing .wav files.\n"
        )
        sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()

    # ── Resolve & validate ───────────────────────────────────────────────────
    mode = _resolve_mode(args)
    ckpt = _resolve_ckpt_paths(args)
    _validate_paths(args, ckpt, mode)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    # ── Banner ───────────────────────────────────────────────────────────────
    print()
    _log("=" * 52)
    _log("  Real-Time Collaborative Cloner  (Phase 1)")
    _log("=" * 52)
    _log(f"Mode            : {'TEXT → CLONED VOICE (TTS)' if mode == 'tts' else 'AUDIO → CLONED VOICE (VC)'}")
    _log(f"Reference audio : {args.reference_audio}")
    _log(f"Output          : {args.output}")
    if mode == "tts":
        _log(f"Text            : {args.text[:80]}{'…' if len(args.text) > 80 else ''}")
        _log(f"Speaker style   : {args.speaker}  |  Speed: {args.speed}  |  Language: {args.language}")
        _log(f"Chunk size      : {args.chunk_size} words")
    else:
        _log(f"Input audio     : {args.input_audio}")
    _log(f"Real-time play  : {'yes (sounddevice)' if args.play else 'no'}")
    _log(f"Warmup          : {'yes' if args.warmup else 'no'}")
    print()

    # ── Load models (once — cached inside VoiceCloner) ───────────────────────
    t_load = time.perf_counter()
    from voice_cloner import VoiceCloner

    cloner = VoiceCloner(
        tts_config_path=ckpt["tts_config_path"],
        tts_ckpt_path=ckpt["tts_ckpt_path"],
        conv_config_path=ckpt["conv_config_path"],
        conv_ckpt_path=ckpt["conv_ckpt_path"],
        openvoice_dir=args.openvoice_dir,
    )
    _log(f"Model load time : {time.perf_counter() - t_load:.1f}s")

    # ── Reference speaker embedding ──────────────────────────────────────────
    _log("Extracting speaker embedding …")
    cloner.set_reference(args.reference_audio)

    # ── Optional warmup ──────────────────────────────────────────────────────
    if args.warmup:
        cloner.warmup()

    # ── Run pipeline ─────────────────────────────────────────────────────────
    t_gen = time.perf_counter()

    if mode == "tts":
        cloner.clone_from_text(
            text=args.text,
            output_path=args.output,
            speaker_key=args.speaker,
            speed=args.speed,
            language=args.language,
            chunk_size=args.chunk_size,
            play=args.play,
        )
    else:
        cloner.clone_from_audio(
            input_audio_path=args.input_audio,
            output_path=args.output,
            play=args.play,
        )

    elapsed = time.perf_counter() - t_gen
    _log(f"Generation time : {elapsed:.1f}s")
    print()
    print(f"✅  Output saved → {args.output}")
    print()


if __name__ == "__main__":
    main()