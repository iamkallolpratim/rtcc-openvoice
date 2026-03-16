#!/usr/bin/env python3
"""
rtcc_cli.py
-----------
Real-Time Collaborative Cloner – minimal CLI prototype.

Usage
-----
Text → Cloned voice:
    python rtcc_cli.py \
        --reference_audio speaker.wav \
        --text "Hello, this is my cloned voice." \
        --output output.wav

Audio → Cloned voice (voice conversion):
    python rtcc_cli.py \
        --reference_audio speaker.wav \
        --input_audio speech.wav \
        --output output.wav

Optional model-path overrides (if checkpoints are in a non-default location):
    python rtcc_cli.py \
        --reference_audio speaker.wav \
        --text "Hello" \
        --output output.wav \
        --openvoice_dir /path/to/OpenVoice \
        --tts_config   /path/to/base_speakers/EN/config.json \
        --tts_ckpt     /path/to/base_speakers/EN/checkpoint.pth \
        --conv_config  /path/to/converter/config.json \
        --conv_ckpt    /path/to/converter/checkpoint.pth
"""

import argparse
import os
import sys


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="rtcc_cli",
        description="RTCC – Real-Time Collaborative Cloner (minimal CLI prototype)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── Required ────────────────────────────────────────────────────────────
    p.add_argument(
        "--reference_audio",
        required=True,
        metavar="PATH",
        help="Path to the reference .wav file whose voice will be cloned.",
    )
    p.add_argument(
        "--output",
        required=True,
        metavar="PATH",
        help="Path for the output .wav file.",
    )

    # ── Mode: text OR audio (mutually exclusive, one required) ───────────────
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--text",
        metavar="TEXT",
        help="Text to synthesise using the cloned voice.",
    )
    mode.add_argument(
        "--input_audio",
        metavar="PATH",
        help="Path to source audio whose *content* will be converted to the cloned voice.",
    )

    # ── Optional: model paths ────────────────────────────────────────────────
    p.add_argument(
        "--openvoice_dir",
        default="./OpenVoice",
        metavar="PATH",
        help="Root directory of the cloned OpenVoice repository (default: ./OpenVoice).",
    )
    p.add_argument(
        "--tts_config",
        default=None,
        metavar="PATH",
        help="Override path to BaseSpeakerTTS config.json.",
    )
    p.add_argument(
        "--tts_ckpt",
        default=None,
        metavar="PATH",
        help="Override path to BaseSpeakerTTS checkpoint.pth.",
    )
    p.add_argument(
        "--conv_config",
        default=None,
        metavar="PATH",
        help="Override path to ToneColorConverter config.json.",
    )
    p.add_argument(
        "--conv_ckpt",
        default=None,
        metavar="PATH",
        help="Override path to ToneColorConverter checkpoint.pth.",
    )

    # ── Optional: TTS tweaks ─────────────────────────────────────────────────
    p.add_argument(
        "--speaker",
        default="default",
        metavar="KEY",
        help="Base-speaker key for TTS (e.g. 'default', 'whispering'). Default: 'default'.",
    )
    p.add_argument(
        "--speed",
        type=float,
        default=1.0,
        metavar="FLOAT",
        help="Speaking speed multiplier for TTS (default: 1.0).",
    )
    p.add_argument(
        "--language",
        default="English",
        metavar="LANG",
        help="Language for TTS (default: 'English').",
    )

    return p


# ---------------------------------------------------------------------------
# Checkpoint path resolution helpers
# ---------------------------------------------------------------------------

def _resolve_ckpt_paths(args: argparse.Namespace) -> dict:
    """
    Return a dict of resolved checkpoint paths, applying user overrides when
    provided, otherwise constructing defaults relative to openvoice_dir.
    """
    base_dir  = os.path.join(args.openvoice_dir, "openvoice", "checkpoints", "base_speakers", "EN")
    conv_dir  = os.path.join(args.openvoice_dir, "openvoice", "checkpoints", "converter")

    return {
        "tts_config_path":  args.tts_config  or os.path.join(base_dir,  "config.json"),
        "tts_ckpt_path":    args.tts_ckpt    or os.path.join(base_dir,  "checkpoint.pth"),
        "conv_config_path": args.conv_config or os.path.join(conv_dir, "config.json"),
        "conv_ckpt_path":   args.conv_ckpt   or os.path.join(conv_dir, "checkpoint.pth"),
    }


def _validate_paths(args: argparse.Namespace, ckpt: dict) -> None:
    """Fail fast with clear messages if required files are missing."""
    errors: list[str] = []

    if not os.path.isfile(args.reference_audio):
        errors.append(f"  reference_audio not found : {args.reference_audio}")

    if args.input_audio and not os.path.isfile(args.input_audio):
        errors.append(f"  input_audio not found     : {args.input_audio}")

    for label, path in ckpt.items():
        if not os.path.isfile(path):
            errors.append(f"  model file not found ({label}): {path}")

    if errors:
        print("\n[rtcc_cli] ERROR – missing files:\n" + "\n".join(errors))
        print(
            "\nPlease ensure:\n"
            "  1. OpenVoice is cloned to the --openvoice_dir path.\n"
            "  2. Checkpoints are downloaded (see README.md).\n"
            "  3. All audio paths point to existing .wav files.\n"
        )
        sys.exit(1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()

    ckpt = _resolve_ckpt_paths(args)
    _validate_paths(args, ckpt)

    # Ensure output directory exists
    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)

    # ── Import and initialise VoiceCloner ────────────────────────────────────
    from voice_cloner import VoiceCloner

    cloner = VoiceCloner(
        tts_config_path=ckpt["tts_config_path"],
        tts_ckpt_path=ckpt["tts_ckpt_path"],
        conv_config_path=ckpt["conv_config_path"],
        conv_ckpt_path=ckpt["conv_ckpt_path"],
        openvoice_dir=args.openvoice_dir,
    )

    # ── Register reference speaker ───────────────────────────────────────────
    cloner.set_reference(args.reference_audio)

    # ── Run selected pipeline ─────────────────────────────────────────────────
    if args.text:
        print(f"\n[rtcc_cli] Mode: TEXT → CLONED VOICE")
        print(f"           Text    : {args.text[:80]}")
        print(f"           Speaker : {args.speaker}")
        print(f"           Speed   : {args.speed}")
        print(f"           Language: {args.language}\n")

        cloner.clone_from_text(
            text=args.text,
            output_path=args.output,
            speaker_key=args.speaker,
            speed=args.speed,
            language=args.language,
        )

    else:  # args.input_audio
        print(f"\n[rtcc_cli] Mode: AUDIO → CLONED VOICE (voice conversion)")
        print(f"           Input  : {args.input_audio}\n")

        cloner.clone_from_audio(
            input_audio_path=args.input_audio,
            output_path=args.output,
        )

    print(f"\n✅  Output saved to: {args.output}")


if __name__ == "__main__":
    main()
