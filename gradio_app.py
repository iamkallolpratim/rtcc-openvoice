"""
gradio_app.py — Optional browser UI for RTCC.

Install:  pip install gradio
Run:      python gradio_app.py
"""

import tempfile
import gradio as gr

try:
    from voice_cloner import VoiceCloner
except ImportError as e:
    raise SystemExit(
        "VoiceCloner not found. Run this file from the RTCC project root.\n"
        f"Original error: {e}"
    )

# ---------------------------------------------------------------------------
# Lazy-load the model once (shared across all UI calls)
# ---------------------------------------------------------------------------
_cloner: VoiceCloner | None = None


def _get_cloner() -> VoiceCloner:
    global _cloner
    if _cloner is None:
        _cloner = VoiceCloner()
    return _cloner


# ---------------------------------------------------------------------------
# Core handler — wired to the Generate button
# ---------------------------------------------------------------------------
def generate(mode: str, reference_audio: str, text: str, input_audio: str) -> str:
    """
    Returns the path to the generated .wav file (Gradio renders it as a player).
    """
    if not reference_audio:
        raise gr.Error("Please upload a reference audio file.")

    cloner = _get_cloner()
    cloner.set_reference(reference_audio)

    out_path = tempfile.mktemp(suffix=".wav")

    if mode == "Text to Voice":
        if not text.strip():
            raise gr.Error("Please enter some text.")
        cloner.clone_from_text(
            text=text,
            output_path=out_path,
        )

    else:  # Audio to Voice
        if not input_audio:
            raise gr.Error("Please upload an input audio file.")
        cloner.clone_from_audio(
            source_audio_path=input_audio,
            output_path=out_path,
        )

    return out_path


# ---------------------------------------------------------------------------
# UI layout
# ---------------------------------------------------------------------------
with gr.Blocks(title="RTCC – Voice Cloner") as demo:
    gr.Markdown("# 🎙️ RTCC — Voice Cloner\nClone any voice in your browser.")

    mode = gr.Dropdown(
        choices=["Text to Voice", "Audio to Voice"],
        value="Text to Voice",
        label="Mode",
    )

    # NOTE: sources=["upload"] is required in Gradio 4+ to get a static file
    # upload widget. Without it the component defaults to microphone/streaming
    # mode, which causes 404s and "Method not implemented" errors on upload.
    reference_audio = gr.Audio(
        sources=["upload"],
        type="filepath",
        label="Reference Audio (voice to clone)",
    )

    text_input = gr.Textbox(
        label="Text",
        placeholder="Enter text to synthesise…",
        lines=3,
    )

    input_audio = gr.Audio(
        sources=["upload"],
        type="filepath",
        label="Input Audio (content to convert)",
        visible=False,
    )

    def toggle_inputs(selected_mode):
        return (
            gr.update(visible=selected_mode == "Text to Voice"),
            gr.update(visible=selected_mode == "Audio to Voice"),
        )

    mode.change(toggle_inputs, inputs=mode, outputs=[text_input, input_audio])

    btn = gr.Button("🔊 Generate", variant="primary")
    output_audio = gr.Audio(label="Output", autoplay=True)

    btn.click(
        fn=generate,
        inputs=[mode, reference_audio, text_input, input_audio],
        outputs=output_audio,
    )


if __name__ == "__main__":
    demo.launch(share=True)
