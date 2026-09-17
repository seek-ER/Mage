"""Gradio app for Mage-VL — video upload and testing.

    python mage_vl/app.py                       # serve on 0.0.0.0:7860
    python mage_vl/app.py --share --port 7861

Upload a video and ask questions about its content. Supports two video backends:
  - **frames**: Uniform frame sampling using OpenCV
  - **codec**: Traditional (HEVC/H.264) or neural (DCVC-RT) codec processing
"""
from __future__ import annotations

import argparse
import os
import time

import gradio as gr
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

from mage_vl.inference_base import sample_video, count_visual_tokens

DEVICE = "cuda"
_CACHE: dict[str, tuple] = {}  # model_path -> (model, processor)


def load_model(model_path: str, device: str = DEVICE):
    """Load and cache model and processor."""
    model_path = (model_path or "").strip()
    if not model_path:
        raise gr.Error("No model specified.")

    if model_path not in _CACHE:
        try:
            processor = AutoProcessor.from_pretrained(
                model_path, trust_remote_code=True
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype="auto",
                device_map="auto",
            ).eval()
            _CACHE[model_path] = (model, processor)
        except Exception as e:
            raise gr.Error(f"Failed to load model '{model_path}': {type(e).__name__}: {e}")

    return _CACHE[model_path]


def process_video(
    video_path,
    question,
    model_path,
    backend,
    codec_engine,
    num_frames,
    max_pixels,
    max_new_tokens,
    progress=gr.Progress(track_tqdm=False),
):
    """Process video and return answer, visual tokens, processing time, and sampled frames."""
    if not video_path:
        raise gr.Error("Please upload a video.")
    if not (question or "").strip():
        raise gr.Error("Please enter a question.")

    progress(0.1, desc="Loading model...")
    model, processor = load_model(model_path)

    # Build chat template
    messages = [{"role": "user", "content": [
        {"type": "video"}, {"type": "text", "text": question},
    ]}]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    progress(0.3, desc="Processing video...")

    # Process video based on backend
    sampled_frames = None
    if backend == "codec":
        # Codec backend
        codec_config = {
            "engine": "hevc" if codec_engine == "traditional" else "dcvc-rt",
            "target_canvas": int(num_frames),
            "patch": 16,
        }
        if codec_engine == "neural":
            # Download model if needed for neural codec
            model_path_resolved = model_path
            if not os.path.isdir(model_path):
                from huggingface_hub import snapshot_download
                model_path_resolved = snapshot_download(model_path)
            codec_config["dcvc"] = {
                "pkg_dir": os.path.join(model_path_resolved, "neural_codec"),
                "device": str(model.device),
            }

        inputs = processor(
            text=[text],
            videos=[video_path],
            video_backend="codec",
            max_pixels=int(max_pixels),
            codec_config=codec_config,
            return_tensors="pt",
            padding=True,
        )
    else:
        # Frames backend
        sampled_frames = sample_video(video_path, int(num_frames))
        inputs = processor(
            text=[text],
            videos=[sampled_frames],
            return_tensors="pt",
            padding=True,
        )

    # Move inputs to device
    inputs = {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)

    # Count visual tokens
    visual_tokens = count_visual_tokens(inputs)

    progress(0.6, desc="Generating response...")

    # Generate with timing
    with torch.inference_mode():
        torch.cuda.synchronize(model.device)
        start = time.perf_counter()
        output = model.generate(
            **inputs, max_new_tokens=int(max_new_tokens), do_sample=False
        )
        torch.cuda.synchronize(model.device)
        elapsed = time.perf_counter() - start

    # Decode answer
    answer = processor.tokenizer.decode(
        output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()

    progress(1.0, desc="Done")

    return answer, visual_tokens, elapsed, sampled_frames or []


def clear_output():
    """Clear all output fields."""
    return "", 0, 0.0, []


def build_ui():
    """Build Gradio interface."""
    with gr.Blocks(title="Mage-VL Video Testing") as demo:
        gr.Markdown("# Mage-VL 视频测试\n上传视频并提问，测试视频理解能力。")

        with gr.Row():
            # Left column: Configuration
            with gr.Column(scale=1):
                gr.Markdown("### 模型配置")
                model_dropdown = gr.Dropdown(
                    choices=["microsoft/Mage-VL"],
                    value="microsoft/Mage-VL",
                    label="模型",
                )
                custom_model = gr.Textbox(
                    label="自定义模型（可选）",
                    placeholder="Hugging Face repo id 或本地路径",
                    lines=1,
                )

                backend_radio = gr.Radio(
                    choices=["frames", "codec"],
                    value="frames",
                    label="视频后端",
                    info="frames: 均匀采样帧；codec: 编解码器处理",
                )
                codec_engine = gr.Dropdown(
                    choices=["traditional", "neural"],
                    value="traditional",
                    label="编解码引擎（仅 codec 后端）",
                    info="traditional: HEVC/H.264；neural: DCVC-RT",
                    visible=False,
                )

                with gr.Row():
                    num_frames = gr.Slider(
                        minimum=1, maximum=128, value=32, step=1,
                        label="帧数",
                        info="采样帧数或 codec 目标画布",
                    )
                    max_pixels = gr.Slider(
                        minimum=50000, maximum=500000, value=150000, step=10000,
                        label="最大像素数（codec 后端）",
                    )

                max_new_tokens = gr.Slider(
                    minimum=64, maximum=1024, value=256, step=64,
                    label="最大生成 token 数",
                )

            # Right column: Video input
            with gr.Column(scale=1):
                gr.Markdown("### 视频输入")
                video_input = gr.Video(
                    label="上传视频",
                    sources=["upload"],
                    type="filepath",
                )
                question_input = gr.Textbox(
                    label="问题",
                    value="Describe this video.",
                    lines=2,
                )

                with gr.Row():
                    analyze_btn = gr.Button("分析视频", variant="primary", scale=2)
                    clear_btn = gr.Button("清除", scale=1)

        # Results section
        with gr.Column():
            gr.Markdown("### 结果")
            result_text = gr.Textbox(
                label="模型回答",
                lines=8,
                interactive=False,
            )
            with gr.Row():
                token_count = gr.Number(label="Visual Tokens", precision=0)
                processing_time = gr.Number(label="处理时间（秒）", precision=2)

            frames_gallery = gr.Gallery(
                label="采样帧（仅 frames 后端）",
                columns=8,
                rows=2,
                height="auto",
            )

        # Example videos
        gr.Markdown("### 示例")
        gr.Examples(
            examples=[
                [os.path.join(os.path.dirname(__file__), "assets", "examples", "soccer-broadcast.mp4"),
                 "Describe this video."],
            ],
            inputs=[video_input, question_input],
        )

        # Event handlers
        # Show/hide codec engine dropdown based on backend selection
        backend_radio.change(
            lambda x: gr.update(visible=(x == "codec")),
            inputs=backend_radio,
            outputs=codec_engine,
        )

        # Analyze button
        def get_model_path(model_dropdown, custom_model):
            """Get model path from dropdown or custom input."""
            return (custom_model or "").strip() or model_dropdown

        def analyze_video(
            video_path, question, model_dropdown, custom_model,
            backend, codec_eng, num_frames_val, max_pixels_val, max_new_tokens_val,
            progress=gr.Progress(track_tqdm=False)
        ):
            """Wrapper to handle model path resolution."""
            model_path = get_model_path(model_dropdown, custom_model)
            return process_video(
                video_path, question, model_path, backend, codec_eng,
                num_frames_val, max_pixels_val, max_new_tokens_val, progress
            )

        analyze_btn.click(
            fn=clear_output,
            inputs=None,
            outputs=[result_text, token_count, processing_time, frames_gallery],
        ).then(
            fn=analyze_video,
            inputs=[
                video_input,
                question_input,
                model_dropdown,
                custom_model,
                backend_radio,
                codec_engine,
                num_frames,
                max_pixels,
                max_new_tokens,
            ],
            outputs=[result_text, token_count, processing_time, frames_gallery],
        )

        # Clear button
        clear_btn.click(
            fn=clear_output,
            inputs=None,
            outputs=[result_text, token_count, processing_time, frames_gallery],
        )

    return demo


def main():
    global DEVICE
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda", help="Device (cuda or cpu)")
    ap.add_argument("--host", default="0.0.0.0", help="Server host")
    ap.add_argument("--port", type=int, default=7860, help="Server port")
    ap.add_argument("--share", action="store_true", help="Create public link")
    ap.add_argument(
        "--preload",
        default=None,
        help="Comma-separated model paths to load at startup",
    )
    args = ap.parse_args()

    DEVICE = args.device

    # Preload models if specified
    if args.preload:
        for model_path in args.preload.split(","):
            load_model(model_path.strip())

    # Build and launch UI
    demo = build_ui()
    demo.queue().launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        theme=gr.themes.Soft(),
    )


if __name__ == "__main__":
    main()
