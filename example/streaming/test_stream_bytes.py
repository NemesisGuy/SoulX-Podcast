import torch
import math
import os
import tempfile
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from gradio_app import infer_from_ui, MODEL_CACHE

# Create a dummy model that returns a short generated wav
class DummyModel:
    def forward_longform(self, **kwargs):
        # produce a 1s 24000Hz sine wave at 440Hz
        sr = 24000
        t = torch.arange(0, 1*sr, dtype=torch.float32) / sr
        wav = 0.5 * torch.sin(2 * math.pi * 440 * t)
        # shape should be (1, samples)
        wav = wav.unsqueeze(0)
        return {"generated_wavs": [wav]}

# monkeypatch MODEL_CACHE to avoid loading real models
MODEL_CACHE["model_path"] = "dummy"
MODEL_CACHE["llm_engine"] = "hf"
MODEL_CACHE["fp16_flow"] = False
MODEL_CACHE["seed"] = 42
MODEL_CACHE["model"] = DummyModel()
MODEL_CACHE["dataset"] = object()

# Call infer_from_ui with streaming enabled and request bytes
gen = infer_from_ui(
    json_file=None,
    manual_text="[S1]Test",
    prompt_wav_file=None,
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
    output_path="outputs/test_stream_bytes.wav",
    llm_engine="hf",
    fp16_flow=False,
    seed=42,
    sample_choice_s2=None,
    use_sample_s2=False,
    auto_fill_s2=False,
    stream_enabled=True,
    stream_as_bytes=True,
)

print("Outputs from streaming-with-bytes test:")
for out in gen:
    print(type(out), len(out) if hasattr(out, '__len__') else None)
    try:
        if len(out) == 4:
            status, path, model, pcm = out
            print('status:', status)
            print('path:', path)
            print('model:', model)
            print('pcm bytes len:', len(pcm) if pcm is not None else None)
        else:
            print(out)
    except Exception as e:
        print('error unpacking', e)
print('done')
