import torch
import math
import os
import tempfile
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from gradio_app import infer_from_ui, MODEL_CACHE

# Create a dummy model that returns a short generated wav
class DummyModel:
    def forward_longform(self, **kwargs):
        # produce a 2s 24000Hz sine wave at 440Hz
        sr = 24000
        t = torch.arange(0, 2*sr, dtype=torch.float32) / sr
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

# Call infer_from_ui with streaming enabled
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
    output_path="outputs/test_monkey_stream.wav",
    llm_engine="hf",
    fp16_flow=False,
    seed=42,
    sample_choice_s2=None,
    use_sample_s2=False,
    auto_fill_s2=False,
    stream_enabled=True,
)

print("Outputs from monkeypatched streaming run:")
for out in gen:
    print(out)
    # if an audio path is yielded, check file exists and size
    try:
        status, audio_path, model_path = out
        if audio_path:
            print("Audio exists:", os.path.exists(audio_path), "size:", os.path.getsize(audio_path))
    except Exception:
        pass
print("Done")
