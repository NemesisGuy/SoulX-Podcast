"""Smoke test for streaming mode.

This quick script calls the UI inference function directly and exercises the
streaming code path. It does not start the Gradio server. Run from project
root as:

python example/streaming/smoke_test.py

"""
import tempfile
import torch
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from gradio_app import infer_from_ui


def make_silent_wav(path, sr=24000, duration_s=0.5):
    import soundfile as sf

    samples = int(sr * duration_s)
    data = torch.zeros((samples,), dtype=torch.float32).numpy()
    sf.write(path, data, sr)


def run():
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    make_silent_wav(tmp.name)

    # Minimal inputs to infer_from_ui: use manual text and a silent prompt wav
    gen = infer_from_ui(
        json_file=None,
        manual_text="[S1]Hello from smoke test",
        prompt_wav_file=tmp.name,
        sample_choice_s1=None,
        use_sample_s1=False,
        prompt_text="",
        speaker2_prompt_wav_file=None,
        speaker2_prompt_text="",
        dialogue_text="",
        use_dialect_prompt=False,
        dialect_prompt_text="",
        model_choice=None,
        model_path=None,
        output_path="outputs/stream_smoke.wav",
        llm_engine="hf",
        fp16_flow=False,
        seed=42,
        sample_choice_s2=None,
        use_sample_s2=False,
        auto_fill_s2=False,
        stream_enabled=True,
    )

    # infer_from_ui is a generator when stream_enabled=True
    print("Streaming outputs:")
    try:
        for out in gen:
            print(out)
    except TypeError:
        print("Function did not yield — streaming not active")


if __name__ == "__main__":
    run()
