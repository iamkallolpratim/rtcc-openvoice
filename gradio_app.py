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

_cloner: VoiceCloner | None = None

def _get_cloner() -> VoiceCloner:
    global _cloner
    if _cloner is None:
        _cloner = VoiceCloner()
    return _cloner


def generate(mode: str, reference_audio: str, text: str, input_audio: str) -> str:
    if not reference_audio:
        raise gr.Error("Please upload a reference audio file.")

    cloner = _get_cloner()
    cloner.set_reference(reference_audio)
    out_path = tempfile.mktemp(suffix=".wav")

    if mode == "Text to Voice":
        if not text.strip():
            raise gr.Error("Please enter some text.")
        cloner.clone_from_text(text=text, output_path=out_path)
    else:
        if not input_audio:
            raise gr.Error("Please upload an input audio file.")
        cloner.clone_from_audio(input_audio_path=input_audio, output_path=out_path)

    return out_path


with gr.Blocks(title="RTCC – Voice Cloner") as demo:
    gr.Markdown("# 🎙️ RTCC — Voice Cloner\nClone any voice in your browser.")

    mode = gr.Dropdown(
        choices=["Text to Voice", "Audio to Voice"],
        value="Text to Voice",
        label="Mode",
    )

    reference_audio = gr.Audio(
        sources=["upload"], type="filepath",
        label="Reference Audio (voice to clone)",
    )

    # Both start visible=True — Gradio only reliably passes values for
    # components that were visible when the page first loaded.
    # We hide the inactive one immediately via JS after load (see demo.load).
    text_input = gr.Textbox(
        label="Text", placeholder="Enter text to synthesise…",
        lines=3, visible=True,
    )
    input_audio = gr.Audio(
        sources=["upload"], type="filepath",
        label="Input Audio (content to convert)",
        visible=True,   # must start True — see note above
    )

    def sync_visibility(selected_mode):
        is_text = selected_mode == "Text to Voice"
        return gr.update(visible=is_text), gr.update(visible=not is_text)

    # Hide the right field on dropdown change AND on initial page load
    mode.change(sync_visibility, inputs=mode, outputs=[text_input, input_audio])
    demo.load(sync_visibility, inputs=mode, outputs=[text_input, input_audio])

    btn = gr.Button("🔊 Generate", variant="primary")
    output_audio = gr.Audio(label="Output", autoplay=True)

    btn.click(
        fn=generate,
        inputs=[mode, reference_audio, text_input, input_audio],
        outputs=output_audio,
    )

if __name__ == "__main__":
    demo.launch()