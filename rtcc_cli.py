#!/usr/bin/env python3
"""
rtcc_cli.py  (Phase 2 — Multi-Engine)
--------------------------------------
Real-Time Collaborative Cloner — command-line interface.

Supports three engines:
  openvoice   — Original OpenVoice V2 (default, backward-compatible)
  chatterbox  — Chatterbox TTS/VC (high-quality, emotion control)

═══════════════════════════════════════════════════════════════════
BACKWARD-COMPATIBLE — all Phase 1 commands work unchanged:
═══════════════════════════════════════════════════════════════════

  python rtcc_cli.py \\
      --reference_audio speaker.wav \\
      --text "Hello" \\
      --output outputs/result.wav
═══════════════════════════════════════════════════════════════════
NEW: High-quality TTS with Chatterbox Turbo:
═══════════════════════════════════════════════════════════════════

  python rtcc_cli.py \\
      --engine chatterbox \\
      --reference_audio speaker.wav \\
      --text "Hi there [laugh], how are you?" \\
      --output outputs/result.wav

═══════════════════════════════════════════════════════════════════
NEW: Chatterbox voice conversion:
═══════════════════════════════════════════════════════════════════

  python rtcc_cli.py \\
      --engine chatterbox \\
      --cb_variant vc \\
      --reference_audio speaker.wav \\
      --input_audio speech.wav \\
      --output outputs/converted.wav
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
        description="RTCC – Real-Time Collaborative Cloner  (Phase 2: Multi-Engine)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── Engine selector (NEW — defaults to openvoice for backward compat) ───
    p.add_argument(
        "--engine",
        choices=["openvoice", "chatterbox"],
        default="openvoice",
        help=(
            "Voice cloning engine.  "
            "'openvoice' = original OpenVoice V2 (default).  "
            "'chatterbox' = Chatterbox TTS/VC (350-500M params).  "
        ),
    )

    # ── Mode ─────────────────────────────────────────────────────────────────
    p.add_argument(
        "--mode",
        choices=["tts", "vc", "rtvc"],
        default=None,
        help=(
            "Pipeline mode.  "
            "'tts' = text → cloned speech.  "
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
        default=None,
        metavar="PATH",
        help="Output .wav file path (optional for rtvc mode).",
    )

    # ── Input (text or audio — one required for tts/vc) ──────────────────────
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

    # ── Streaming / playback ─────────────────────────────────────────────────
    p.add_argument(
        "--play",
        action="store_true",
        default=False,
        help="Stream audio to speakers during generation.",
    )
    p.add_argument(
        "--chunk_size",
        type=int,
        default=15,
        metavar="N",
        help="[TTS mode] Max words per synthesis chunk (default: 15).",
    )

    # ── Warmup ───────────────────────────────────────────────────────────────
    p.add_argument(
        "--warmup",
        action="store_true",
        default=False,
        help="Run a dummy inference pass to pre-heat model caches.",
    )

    # ── TTS style (OpenVoice) ────────────────────────────────────────────────
    p.add_argument(
        "--speaker",
        default="default",
        metavar="KEY",
        help="Base speaker style (OpenVoice). Choices: default, whispering, etc.",
    )
    p.add_argument(
        "--speed",
        type=float,
        default=1.0,
        metavar="FLOAT",
        help="Speaking rate multiplier (default: 1.0).",
    )
    p.add_argument(
        "--language",
        default="English",
        metavar="LANG",
        help="TTS language string (default: English).",
    )

    # ── Model paths (OpenVoice overrides) ────────────────────────────────────
    p.add_argument("--openvoice_dir", default="./OpenVoice", metavar="PATH",
                   help="Root of the cloned OpenVoice repo.")
    p.add_argument("--tts_config",  default=None, metavar="PATH")
    p.add_argument("--tts_ckpt",    default=None, metavar="PATH")
    p.add_argument("--conv_config", default=None, metavar="PATH")
    p.add_argument("--conv_ckpt",   default=None, metavar="PATH")


    p.add_argument(
        "--rt_chunk_ms",
        type=int,
        default=200,
        metavar="MS",
        help="[rtvc mode] Real-time chunk size in ms (default: 200).",
    )
    p.add_argument(
        "--steps",
        type=int,
        default=2,
        metavar="N",
    )

    # ── Chatterbox options (NEW) ─────────────────────────────────────────────
    p.add_argument(
        "--cb_variant",
        choices=["standard", "turbo", "vc"],
        default="turbo",
        help="Chatterbox model variant (default: turbo).",
    )
    p.add_argument(
        "--exaggeration",
        type=float,
        default=0.5,
        metavar="FLOAT",
        help="[Chatterbox standard] Emotion exaggeration 0.0–1.0+ (default: 0.5).",
    )
    p.add_argument(
        "--cfg_weight",
        type=float,
        default=0.5,
        metavar="FLOAT",
        help="[Chatterbox standard] CFG weight 0.0–1.0 (default: 0.5).",
    )

    return p


# ─────────────────────────────────────────────────────────────────────────────
# Mode resolution
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_mode(args: argparse.Namespace) -> str:
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
        "(Or set --mode tts / --mode vc / --mode rtvc explicitly.)"
    )


def _abort(msg: str) -> None:
    print(f"\n[RTCC] ERROR — {msg}\n", file=sys.stderr)
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# OpenVoice checkpoint resolution (unchanged from Phase 1)
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_ckpt_paths(args: argparse.Namespace) -> dict[str, str]:
    base = os.path.join(args.openvoice_dir, "openvoice", "checkpoints",
                        "base_speakers", "EN")
    converter = os.path.join(args.openvoice_dir, "openvoice", "checkpoints",
                             "converter")
    return {
        "tts_config_path":  args.tts_config or os.path.join(base,      "config.json"),
        "tts_ckpt_path":    args.tts_ckpt or os.path.join(base,      "checkpoint.pth"),
        "conv_config_path": args.conv_config or os.path.join(converter, "config.json"),
        "conv_ckpt_path":   args.conv_ckpt or os.path.join(converter, "checkpoint.pth"),
    }


def _validate_paths(args: argparse.Namespace, ckpt: dict[str, str], mode: str) -> None:
    errors: list[str] = []

    if not os.path.isfile(args.reference_audio):
        errors.append(f"  reference_audio not found  : {args.reference_audio}")

    if mode == "vc" and args.input_audio and not os.path.isfile(args.input_audio):
        errors.append(f"  input_audio not found      : {args.input_audio}")

    # Only validate OpenVoice checkpoints for the openvoice engine
    if args.engine == "openvoice":
        for label, path in ckpt.items():
            if not os.path.isfile(path):
                errors.append(f"  model file not found ({label}): {path}")

    if errors:
        print("\n[RTCC] ERROR – missing files:\n" + "\n".join(errors))
        print(
            "\nPlease ensure:\n"
            "  1. The engine's repo is cloned and models downloaded.\n"
            "  2. All audio paths point to existing .wav files.\n"
        )
        sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Engine dispatchers
# ─────────────────────────────────────────────────────────────────────────────

def _run_openvoice(args, mode, ckpt):
    """Original OpenVoice pipeline — fully backward-compatible."""
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

    cloner.set_reference(args.reference_audio)

    if args.warmup:
        cloner.warmup()

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


def _run_chatterbox(args, mode):
    """Chatterbox pipeline — TTS and VC."""
    from chatterbox_engine import ChatterboxEngine

    # Auto-select variant: use 'vc' variant for VC mode
    variant = args.cb_variant
    if mode == "vc" and variant != "vc":
        _log(f"Switching to Chatterbox VC variant for voice conversion")
        variant = "vc"

    engine = ChatterboxEngine(variant=variant)
    engine.load_models()
    engine.set_reference(args.reference_audio)

    if mode == "tts":
        if not args.output:
            _abort("--output is required for TTS mode")
        engine.clone_from_text(
            text=args.text,
            output_path=args.output,
            exaggeration=args.exaggeration,
            cfg_weight=args.cfg_weight,
            play=args.play,
        )
    elif mode == "vc":
        if not args.output:
            _abort("--output is required for VC mode")
        engine.clone_from_audio(
            input_audio_path=args.input_audio,
            output_path=args.output,
            play=args.play,
        )
    elif mode == "rtvc":
        _abort(
            "Chatterbox does not support real-time VC. Use --engine meanvc for rtvc mode.")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # ── Resolve & validate ───────────────────────────────────────────────────
    mode = _resolve_mode(args)

    # Output is required for all modes except rtvc
    if mode != "rtvc" and args.output is None:
        _abort("--output is required for tts and vc modes.")

    ckpt = _resolve_ckpt_paths(args) if args.engine == "openvoice" else {}
    _validate_paths(args, ckpt, mode)

    # Ensure output directory exists
    if args.output:
        os.makedirs(os.path.dirname(
            os.path.abspath(args.output)), exist_ok=True)

    # ── Banner ───────────────────────────────────────────────────────────────
    MODE_LABELS = {
        "tts":  "TEXT → CLONED VOICE (TTS)",
        "vc":   "AUDIO → CLONED VOICE (VC)",
        "rtvc": "REAL-TIME MIC → SPEAKER (RTVC)",
    }
    ENGINE_LABELS = {
        "openvoice":  "OpenVoice V2",
        "chatterbox": f"Chatterbox-{args.cb_variant.title()}",
    }

    print()
    _log("=" * 56)
    _log("  Real-Time Collaborative Cloner  (Phase 2)")
    _log("=" * 56)
    _log(f"Engine          : {ENGINE_LABELS[args.engine]}")
    _log(f"Mode            : {MODE_LABELS[mode]}")
    _log(f"Reference audio : {args.reference_audio}")
    if args.output:
        _log(f"Output          : {args.output}")
    if mode == "tts":
        _log(
            f"Text            : {args.text[:80]}{'…' if len(args.text) > 80 else ''}")
        if args.engine == "openvoice":
            _log(f"Speaker style   : {args.speaker}  |  Speed: {args.speed}")
        elif args.engine == "chatterbox":
            _log(
                f"Exaggeration    : {args.exaggeration}  |  CFG: {args.cfg_weight}")
    elif mode == "vc":
        _log(f"Input audio     : {args.input_audio}")
    elif mode == "rtvc":
        _log(f"RT chunk size   : {args.rt_chunk_ms}ms")
    _log(f"Real-time play  : {'yes' if args.play else 'no'}")
    print()

    # ── Dispatch to engine ───────────────────────────────────────────────────
    t_start = time.perf_counter()

    if args.engine == "openvoice":
        _run_openvoice(args, mode, ckpt)
    elif args.engine == "chatterbox":
        _run_chatterbox(args, mode)

    _log(f"Total time: {time.perf_counter() - t_start:.1f}s")
    _log("Done ✓")


if __name__ == "__main__":
    main()
