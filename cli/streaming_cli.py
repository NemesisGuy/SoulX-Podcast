"""Interactive streaming CLI.

Usage:
  python cli/streaming_cli.py [--model-path PATH] [--llm hf|vllm] [--fp16]

Run the script, then type lines of text and hit Enter. Each line is queued as a streaming job
that calls `infer_from_ui(..., stream_enabled=True)`. Streaming chunks yielded by the
function will be printed and played (Windows only) using winsound.

Notes:
- This script assumes it's run on Windows (uses winsound). If not Windows, it will
  only print the yielded audio paths and not attempt playback.
- The script uses a background worker thread to consume the job queue so you can
  continue typing while jobs run.
"""
from __future__ import annotations

import argparse
import queue
import threading
import time
import os
import sys
from typing import Optional

# ensure project root is on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gradio_app import infer_from_ui

IS_WINDOWS = os.name == "nt"
if IS_WINDOWS:
    try:
        import winsound
    except Exception:
        winsound = None
else:
    winsound = None

# prefer in-memory playback via simpleaudio when available
try:
    import simpleaudio as sa
except Exception:
    sa = None

Job = dict


def worker(job_queue: "queue.Queue[Job]", stop_event: threading.Event):
    """Background worker that consumes queued text inputs and runs streaming inference."""
    while not stop_event.is_set() or not job_queue.empty():
        try:
            job = job_queue.get(timeout=0.5)
        except Exception:
            continue

        text = job.get("text", "")
        params = job.get("params", {})
        print(f"[worker] starting job: {text[:60]!r}")

        # ensure text includes speaker tag
        if not text.startswith("[S"):
            text = f"[S1]{text}"

        try:
            gen = infer_from_ui(
                json_file=None,
                manual_text=text,
                prompt_wav_file=params.get("prompt_wav_file"),
                sample_choice_s1=None,
                use_sample_s1=False,
                prompt_text="",
                speaker2_prompt_wav_file=None,
                speaker2_prompt_text="",
                dialogue_text="",
                use_dialect_prompt=False,
                dialect_prompt_text="",
                model_choice=params.get("model_choice"),
                model_path=params.get("model_path"),
                output_path=params.get("output_path", "outputs/cli_stream_output.wav"),
                llm_engine=params.get("llm_engine", "hf"),
                fp16_flow=params.get("fp16_flow", False),
                seed=params.get("seed", 42),
                sample_choice_s2=None,
                use_sample_s2=False,
                auto_fill_s2=False,
                stream_enabled=True,
                stream_as_bytes=True,
            )

            for out in gen:
                # out may be (status, audio_path, model_path) or (status, audio_path, model_path, pcm_bytes)
                status = None
                audio_path = None
                model_path = None
                pcm_bytes = None
                try:
                    if len(out) == 4:
                        status, audio_path, model_path, pcm_bytes = out
                    elif len(out) == 3:
                        status, audio_path, model_path = out
                except Exception:
                    status = str(out)

                print(f"[worker] status: {status}")
                if pcm_bytes is not None and sa is not None:
                    try:
                        # assume mono 24000Hz, 16-bit PCM
                        sr = 24000
                        num_channels = 1
                        bytes_per_sample = 2
                        play_obj = sa.play_buffer(pcm_bytes, num_channels, bytes_per_sample, sr)
                        # block until this chunk finishes to avoid overlap/echo
                        play_obj.wait_done()
                    except Exception as e:
                        print(f"[worker] simpleaudio play failed: {e}")
                        pcm_bytes = None

                if pcm_bytes is None:
                    if audio_path:
                        print(f"[worker] audio file: {audio_path}")
                        # try to play on Windows using winsound
                        if IS_WINDOWS and winsound is not None:
                            try:
                                # Play synchronously (blocking) to avoid overlapping chunks
                                winsound.PlaySound(audio_path, winsound.SND_FILENAME)
                            except Exception as e:
                                print(f"[worker] could not play audio: {e}")

            print(f"[worker] job finished: {text[:60]!r}")
        except Exception as e:
            print(f"[worker] job failed: {e}")
        finally:
            job_queue.task_done()


def main(argv: Optional[list[str]] = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default=None, help="Local model directory (optional)")
    parser.add_argument("--llm", type=str, default="hf", help="LM engine to use: hf or vllm")
    parser.add_argument("--fp16", action="store_true", help="Use fp16 for flow model")
    parser.add_argument("--output", type=str, default="outputs/cli_stream_output.wav", help="Base output path")
    parser.add_argument("--prompt-wav", type=str, default=None, help="Path to reference WAV to use as prompt for speaker style")
    parser.add_argument("--sample", type=str, default=None, help="Name of sample under example/audios to use as prompt (e.g. female_mandarin.wav)")
    args = parser.parse_args(argv)

    q: "queue.Queue[Job]" = queue.Queue()
    stop_event = threading.Event()
    t = threading.Thread(target=worker, args=(q, stop_event), daemon=True)
    t.start()

    print("Interactive streaming CLI. Type lines (Enter to queue). Type /quit or /q to exit.")
    print("Runtime commands: /sample <path|name> to change reference sample, /prompt <path> to set prompt wav")
    # Resolve initial prompt wav
    current_prompt = None
    if args.sample:
        sample_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "example", "audios", args.sample)
        if os.path.exists(sample_path):
            current_prompt = sample_path
        else:
            print(f"Warning: sample not found: {sample_path}")
    if args.prompt_wav:
        if os.path.exists(args.prompt_wav):
            current_prompt = args.prompt_wav
        else:
            print(f"Warning: prompt wav not found: {args.prompt_wav}")
    try:
        while True:
            try:
                line = input("> ")
            except EOFError:
                break
            if not line:
                continue
            if line.strip().lower() in ("/quit", "/q", "quit", "exit"):
                break

            # runtime command to change sample/prompt
            if line.strip().startswith("/sample "):
                arg = line.strip()[8:].strip()
                # try resolve as a repo sample first
                candidate = os.path.join(os.path.dirname(os.path.dirname(__file__)), "example", "audios", arg)
                if os.path.exists(candidate):
                    current_prompt = candidate
                    print(f"Set current sample to: {current_prompt}")
                elif os.path.exists(arg):
                    current_prompt = arg
                    print(f"Set current sample to: {current_prompt}")
                else:
                    print(f"Sample not found: {arg}")
                continue
            if line.strip().startswith("/prompt "):
                arg = line.strip()[8:].strip()
                if os.path.exists(arg):
                    current_prompt = arg
                    print(f"Set current prompt wav to: {current_prompt}")
                else:
                    print(f"Prompt wav not found: {arg}")
                continue

            job = {
                "text": line,
                "params": {
                    "model_path": args.model_path,
                    "model_choice": None,
                    "output_path": args.output,
                    "prompt_wav_file": current_prompt,
                    "llm_engine": args.llm,
                    "fp16_flow": args.fp16,
                    "seed": 42,
                },
            }
            q.put(job)
            print("[main] queued job")

    finally:
        print("Waiting for queued jobs to finish...")
        stop_event.set()
        q.join()
        print("All jobs finished. Exiting.")


if __name__ == "__main__":
    main()
