#!/usr/bin/env python
"""Run Mage-VL image or video inference offline or through SGLang."""

import argparse
import base64
import mimetypes
from pathlib import Path


def image_url(image: str) -> str:
    if image.startswith(("http://", "https://", "data:")):
        return image
    path = Path(image)
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def sample_video(video: str, num_frames: int):
    import cv2
    import numpy as np
    from PIL import Image

    capture = cv2.VideoCapture(video)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        capture.release()
        raise ValueError(f"Could not read video: {video}")
    indices = np.linspace(0, frame_count - 1, min(num_frames, frame_count), dtype=int)
    frames = []
    for index in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = capture.read()
        if not ok:
            capture.release()
            raise ValueError(f"Could not decode frame {index} from: {video}")
        frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    capture.release()
    return frames


def count_visual_tokens(inputs: dict) -> int:
    """Return total visual token count from processor outputs.

    Uses image_grid_thw when present (Mage-VL codec + frames paths both emit it).
    Falls back to pixel_values shape for other models.
    """
    if "image_grid_thw" in inputs:
        grid = inputs["image_grid_thw"]
        return int(grid.prod(dim=-1).sum().item())
    if "pixel_values" in inputs:
        pv = inputs["pixel_values"]
        if pv.ndim >= 3:
            return int(pv.shape[-2] * pv.shape[-1])
    return 0


def run_offline(args):
    import os
    import time
    import torch
    from PIL import Image
    from transformers import AutoModelForCausalLM, AutoProcessor

    model_path = args.model
    if args.video and args.video_backend == "codec" and args.codec_engine == "neural":
        if not os.path.isdir(model_path):
            from huggingface_hub import snapshot_download

            model_path = snapshot_download(args.model)
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, torch_dtype="auto", device_map="auto"
    ).eval()
    media_type = "image" if args.image else "video"
    messages = [{"role": "user", "content": [
        {"type": media_type}, {"type": "text", "text": args.question},
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if args.image:
        inputs = processor(
            text=[text], images=[Image.open(args.image).convert("RGB")], return_tensors="pt"
        )
    elif args.video_backend == "codec":
        codec_config = {
            "engine": "hevc" if args.codec_engine == "traditional" else "dcvc-rt",
            "target_canvas": args.num_frames,
            "patch": 16,
        }
        if args.codec_engine == "neural":
            codec_config["dcvc"] = {
                "pkg_dir": os.path.join(model_path, "neural_codec"),
                "device": str(model.device),
            }
        inputs = processor(
            text=[text],
            videos=[args.video],
            video_backend="codec",
            max_pixels=args.max_pixels,
            codec_config=codec_config,
            return_tensors="pt",
            padding=True,
        )
    else:
        inputs = processor(
            text=[text],
            videos=[sample_video(args.video, args.num_frames)],
            return_tensors="pt",
            padding=True,
        )
    inputs = {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)

    visual_tokens = count_visual_tokens(inputs)
    backend = "codec" if args.video and args.video_backend == "codec" else "frames"
    if args.image:
        backend = "image"
    print(f"[{backend}] visual_tokens = {visual_tokens}")

    with torch.inference_mode():
        torch.cuda.synchronize(model.device)
        start = time.perf_counter()
        output = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        torch.cuda.synchronize(model.device)
        elapsed = time.perf_counter() - start
    print(f"[{backend}] time = {elapsed:.2f}s")

    answer = processor.tokenizer.decode(
        output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )
    print(answer.strip())


def run_online(args):
    from openai import OpenAI

    if args.image:
        content = [{"type": "image_url", "image_url": {"url": image_url(args.image)}}]
    else:
        content = [
            {"type": "image_url", "image_url": {"url": frame_url(frame)}}
            for frame in sample_video(args.video, args.num_frames)
        ]
    content.append({"type": "text", "text": args.question})
    client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    response = client.chat.completions.create(
        model=args.model,
        messages=[{"role": "user", "content": content}],
        max_tokens=args.max_new_tokens,
    )
    print(response.choices[0].message.content)


def frame_url(frame) -> str:
    import io

    buffer = io.BytesIO()
    frame.save(buffer, format="JPEG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("offline", "online"), required=True)
    media = parser.add_mutually_exclusive_group(required=True)
    media.add_argument("--image", help="Local image path")
    media.add_argument("--video", help="Local video path")
    parser.add_argument("--video-backend", choices=("frames", "codec"), default="frames")
    parser.add_argument(
        "--codec-engine", choices=("traditional", "neural"), default="traditional"
    )
    parser.add_argument("--num-frames", type=int, default=32)
    parser.add_argument("--max-pixels", type=int, default=150000)
    parser.add_argument("--question", default="Describe this media.")
    parser.add_argument("--model", default="microsoft/Mage-VL")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--base-url", default="http://localhost:30000/v1")
    parser.add_argument("--api-key", default="EMPTY")
    args = parser.parse_args()
    if args.mode == "online" and args.video_backend == "codec":
        parser.error("online video inference supports only --video-backend frames")
    if args.num_frames <= 0:
        parser.error("--num-frames must be positive")
    (run_offline if args.mode == "offline" else run_online)(args)


if __name__ == "__main__":
    main()
