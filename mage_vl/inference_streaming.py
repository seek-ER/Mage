"""Run event-gated Mage-VL inference on an arbitrary video.

The video is split into non-overlapping segments. The optional StreamMind gate
scores the complete causal segment stream once, then standard Transformers
generation runs only on each current segment whose gate score crosses the
threshold.
"""
from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

import torch
from decord import VideoReader, cpu
from transformers import AutoModelForCausalLM, AutoProcessor


USER_PROMPT = "Please describe the video content in detail based on the provided information."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--video", required=True, help="Input video path.")
    parser.add_argument("--checkpoint", default="microsoft/Mage-VL")
    parser.add_argument("--video_backend", choices=("codec", "frames"), default="codec")
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--cur_fps", type=float, default=2)
    parser.add_argument("--segment_sec", type=float, default=8.0)
    parser.add_argument("--max_new_tokens", type=int, default=80)
    parser.add_argument("--max_segments", type=int, default=0)
    parser.add_argument("--gate_threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--attn_impl",
        choices=("flash_attention_2", "sdpa", "eager"),
        default="flash_attention_2",
    )
    args = parser.parse_args()
    if args.segment_sec <= 0:
        parser.error("--segment_sec must be greater than 0")
    return args


def to_device(inputs: dict, device: str, dtype: torch.dtype) -> dict:
    return {
        key: (value.to(device=device, dtype=dtype) if key == "pixel_values" else value.to(device))
        for key, value in inputs.items()
    }


def extract_subclip(
    src_video: str, start: float, duration: float, out_path: Path, timeout: int = 120,
) -> Path:
    """Cut [start, start+duration] from src_video without re-encoding."""
    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(f".tmp_{out_path.name}")
    # -c copy seeks to the nearest keyframe, which is within this script's tolerance.
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-ss", f"{max(0.0, start):.3f}",
            "-t", f"{duration:.3f}",
            "-i", str(src_video),
            "-c", "copy",
            "-avoid_negative_ts", "make_zero",
            str(tmp_path),
        ],
        check=True,
        timeout=timeout,
        capture_output=True,
    )
    tmp_path.replace(out_path)
    return out_path


def build_segment_inputs(
    processor,
    video_path: str,
    start: float,
    end: float,
    backend: str,
    clip_dir: Path,
    num_frames: int,
    cur_fps: float,
    prompt: str = USER_PROMPT,
) -> dict | None:
    duration = end - start
    clip_path = clip_dir / (
        f"{Path(video_path).stem}__t{int(start * 100):08d}__d{int(duration * 100):08d}.mp4"
    )
    try:
        extract_subclip(video_path, start, duration, clip_path)
    except Exception as error:
        print(f"[warn] ffmpeg failed for {start:.2f}-{end:.2f}: {error}")
        return None

    messages = [{
        "role": "user",
        "content": [
            {"type": "video"},
            {"type": "text", "text": prompt},
        ],
    }]
    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    kwargs = {
        "text": [prompt],
        "videos": [str(clip_path)],
        "video_backend": backend,
        "return_tensors": "pt",
        "padding": False,
    }
    if backend == "codec":
        kwargs.update(
            codec_config={"patch": 16, "max_pixels": 150000},
            max_pixels=150000,
        )
    else:
        kwargs.update(num_frames=num_frames, target_fps=cur_fps)

    try:
        return processor(**kwargs)
    except Exception as error:
        print(f"[warn] {backend} processor failed for {start:.2f}-{end:.2f}: {error}")
        return None


def gate_inputs(inputs: dict) -> dict:
    return {
        key: inputs[key]
        for key in ("pixel_values", "image_grid_thw", "patch_positions")
        if key in inputs
    }


@torch.inference_mode()
def generate_current_segment(model, processor, inputs: dict, max_new_tokens: int) -> str:
    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    new_tokens = output[0, inputs["input_ids"].shape[1]:]
    return processor.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def main() -> None:
    args = parse_args()
    processor = AutoProcessor.from_pretrained(args.checkpoint, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        attn_implementation=args.attn_impl,
    ).to(args.device).eval()
    if not hasattr(model, "streammind_gate_forward_segments"):
        raise AttributeError("Checkpoint does not expose streammind_gate_forward_segments()")

    reader = VideoReader(args.video, ctx=cpu(0), num_threads=1)
    duration = len(reader) / float(reader.get_avg_fps())
    del reader

    clip_dir = Path(os.environ.get(
        "STREAMMIND_CLIP_CACHE",
        str(Path(args.video).resolve().parent / ".streammind_clips"),
    ))
    clip_dir.mkdir(parents=True, exist_ok=True)
    if args.video_backend == "codec":
        os.environ.setdefault(
            "ONLINE_CODEC_CACHE_DIR",
            str(clip_dir.parent / ".streammind_codec_cache"),
        )

    segments = []
    timestamps = []
    start = 0.0
    segment_index = 0
    while start < duration:
        if args.max_segments and segment_index >= args.max_segments:
            break
        end = min(duration, start + args.segment_sec)
        inputs = build_segment_inputs(
            processor,
            args.video,
            start,
            end,
            args.video_backend,
            clip_dir,
            args.num_frames,
            args.cur_fps,
        )
        if inputs is None:
            print(f"[t={start:.1f}-{end:.1f}s] skip (segment unusable)")
        else:
            segments.append(to_device(inputs, args.device, model.dtype))
            timestamps.append((start, end))
        start = end
        segment_index += 1

    if not segments:
        return

    visual_segments = [gate_inputs(segment) for segment in segments]
    with torch.inference_mode():
        logits = model.streammind_gate_forward_segments(visual_segments)[0]
    lengths = [int(segment["image_grid_thw"][:, 0].sum()) for segment in visual_segments]
    boundaries = torch.tensor(lengths, device=logits.device).cumsum(0) - 1
    probabilities = torch.softmax(logits[boundaries].float(), dim=-1)[:, 1].tolist()

    for (start, end), inputs, probability in zip(
        timestamps, segments, probabilities, strict=True,
    ):
        if probability < args.gate_threshold:
            print(f"[t={start:.1f}-{end:.1f}s] gate=silence (p={probability:.2f})")
            continue
        try:
            text = generate_current_segment(model, processor, inputs, args.max_new_tokens)
        except Exception as error:
            text = f"(generate failed: {error})"
        print(f"[t={start:.1f}-{end:.1f}s] gate=response (p={probability:.2f}) -> {text}")


if __name__ == "__main__":
    main()
