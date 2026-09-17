"""Gradio app for Mage-VL — video upload, QA and event-gated streaming.

    python mage_vl/app.py                       # serve on 0.0.0.0:7860
    python mage_vl/app.py --share --port 7861

Two tabs:
  - **Video QA**: upload a video, ask a question, answer streamed token by token.
    Supports the `frames` (uniform sampling) and `codec` (HEVC/DCVC-RT) backends.
  - **Streaming**: split the video into fixed-length segments, score each with the
    StreamMind cognition gate, and emit commentary only for response-worthy
    segments (event-gated proactive streaming).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

# Allow running as both `python mage_vl/app.py` and `python -m mage_vl.app`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoProcessor, TextIteratorStreamer

from mage_vl.inference_base import sample_video, count_visual_tokens
from mage_vl.inference_streaming import (
    USER_PROMPT,
    build_segment_inputs,
    gate_inputs,
    to_device as segment_to_device,
)

DEVICE = "cuda"
_CACHE: dict[str, tuple] = {}  # model_path -> (model, processor)
_PRELOADED: list[str] = []  # paths loaded at startup, used as UI defaults


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


def build_inputs(
    processor,
    video_path,
    question,
    backend,
    codec_engine,
    num_frames,
    max_pixels,
    model_path,
):
    """Tokenize one video + question pair for the QA path.

    Returns ``(inputs, sampled_frames)``; ``sampled_frames`` is populated only on
    the frames backend so the UI can show what the model actually saw.
    """
    messages = [{"role": "user", "content": [
        {"type": "video"}, {"type": "text", "text": question},
    ]}]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    sampled_frames = None
    if backend == "codec":
        codec_config = {
            "engine": "hevc" if codec_engine == "traditional" else "dcvc-rt",
            "target_canvas": int(num_frames),
            "patch": 16,
        }
        if codec_engine == "neural":
            # The neural codec package ships inside the checkpoint, so a Hub
            # repo id has to be materialized locally first.
            model_path_resolved = model_path
            if not os.path.isdir(model_path):
                from huggingface_hub import snapshot_download
                model_path_resolved = snapshot_download(model_path)
            codec_config["dcvc"] = {
                "pkg_dir": os.path.join(model_path_resolved, "neural_codec"),
                "device": str(DEVICE),
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
        sampled_frames = sample_video(video_path, int(num_frames))
        inputs = processor(
            text=[text],
            videos=[sampled_frames],
            return_tensors="pt",
            padding=True,
        )
    return inputs, sampled_frames


def to_model_device(inputs: dict, model) -> dict:
    inputs = {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)
    return inputs


def _sync(model) -> None:
    """Block until queued kernels finish, so timing is accurate (CUDA only)."""
    try:
        device = model.device
    except Exception:
        return
    if torch.cuda.is_available() and torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


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
    """Answer one question about one video, streaming the answer token by token.

    Yields ``(answer, visual_tokens, elapsed, frames)``; the first yield only
    clears the outputs, later ones append decoded tokens.
    """
    if not video_path:
        raise gr.Error("请先上传视频。")
    if not (question or "").strip():
        raise gr.Error("请输入问题。")

    progress(0.05, desc="加载模型…")
    model, processor = load_model(model_path)

    progress(0.25, desc="处理视频…")
    inputs, sampled_frames = build_inputs(
        processor, video_path, question, backend, codec_engine, num_frames,
        max_pixels, model_path,
    )
    inputs = to_model_device(inputs, model)
    visual_tokens = count_visual_tokens(inputs)

    # Frames are known up front; show them while generation is still running.
    yield "", visual_tokens, 0.0, sampled_frames or []

    progress(0.5, desc="生成回答…")
    streamer = TextIteratorStreamer(
        processor.tokenizer, skip_prompt=True, skip_special_tokens=True
    )
    failure: list[BaseException] = []

    def _generate() -> None:
        try:
            model.generate(
                **inputs,
                max_new_tokens=int(max_new_tokens),
                do_sample=False,
                streamer=streamer,
            )
        except BaseException as error:  # surface it after the loop drains
            failure.append(error)
        finally:
            # Without this the consumer blocks forever whenever generate() raises.
            streamer.end()

    _sync(model)
    start = time.perf_counter()
    thread = threading.Thread(target=_generate, daemon=True)
    thread.start()

    answer = ""
    for chunk in streamer:
        answer += chunk
        yield answer.strip(), visual_tokens, time.perf_counter() - start, sampled_frames or []

    thread.join()
    if failure:
        raise gr.Error(f"生成失败：{type(failure[0]).__name__}: {failure[0]}")
    _sync(model)
    yield answer.strip(), visual_tokens, time.perf_counter() - start, sampled_frames or []


def _video_duration(video_path: str) -> float:
    """Read a video's duration in seconds (decord, falling back to ffprobe)."""
    try:
        from decord import VideoReader, cpu

        reader = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        duration = len(reader) / float(reader.get_avg_fps())
        del reader
        return duration
    except Exception:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                video_path,
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
        return float(result.stdout.strip())


def stream_video(
    video_path,
    model_path,
    backend,
    segment_sec,
    num_frames,
    cur_fps,
    gate_threshold,
    max_new_tokens,
    prompt,
    max_segments,
    progress=gr.Progress(track_tqdm=False),
):
    """Event-gated proactive streaming over a whole video.

    Every segment goes through the cognition gate; only segments scoring at or
    above ``gate_threshold`` are handed to the VLM. Yields the growing log
    so the user sees decisions as they are made.
    """
    if not video_path:
        raise gr.Error("请先上传视频。")

    progress(0.05, desc="加载模型…")
    model, processor = load_model(model_path)
    if not hasattr(model, "streammind_gate_forward_segments"):
        raise gr.Error(
            "当前 checkpoint 未暴露 streammind_gate_forward_segments()，无法使用门控流式分析。"
        )

    try:
        duration = _video_duration(video_path)
    except Exception as e:
        raise gr.Error(f"无法读取视频时长（需要 ffprobe）：{type(e).__name__}: {e}")

    # Clips and codec artifacts are cached per source video, mirroring the CLI.
    clip_dir = Path(tempfile.gettempdir()) / "mage_vl_stream" / Path(video_path).stem
    clip_dir.mkdir(parents=True, exist_ok=True)
    if backend == "codec":
        os.environ.setdefault("ONLINE_CODEC_CACHE_DIR", str(clip_dir.parent / ".codec_cache"))

    progress(0.15, desc="切分视频片段…")
    segments, timestamps = [], []
    start, index = 0.0, 0
    while start < duration:
        if max_segments and index >= int(max_segments):
            break
        end = min(duration, start + float(segment_sec))
        seg_inputs = build_segment_inputs(
            processor, video_path, start, end, backend, clip_dir,
            int(num_frames), float(cur_fps), prompt or USER_PROMPT,
        )
        if seg_inputs is not None:
            segments.append(segment_to_device(seg_inputs, DEVICE, model.dtype))
            timestamps.append((start, end))
        start = end
        index += 1

    if not segments:
        raise gr.Error("没有可用的视频片段，请检查视频格式或 ffmpeg 是否可用。")

    progress(0.4, desc="门控打分…")
    visual_segments = [gate_inputs(segment) for segment in segments]
    with torch.inference_mode():
        logits = model.streammind_gate_forward_segments(visual_segments)[0]
    lengths = [int(segment["image_grid_thw"][:, 0].sum()) for segment in visual_segments]
    boundaries = torch.tensor(lengths, device=logits.device).cumsum(0) - 1
    probabilities = torch.softmax(logits[boundaries].float(), dim=-1)[:, 1].tolist()

    log = (
        f"视频时长 {duration:.1f}s，共 {len(segments)} 段，"
        f"门控阈值 {float(gate_threshold):.2f}\n\n"
    )
    yield log

    for position, ((start, end), seg_inputs, probability) in enumerate(
        zip(timestamps, segments, probabilities, strict=True)
    ):
        progress(
            0.4 + 0.6 * (position + 1) / len(segments),
            desc=f"第 {position + 1}/{len(segments)} 段…",
        )
        if probability < float(gate_threshold):
            log += f"[t={start:.1f}-{end:.1f}s] 🔇 沉默 (p={probability:.2f})\n\n"
            yield log
            continue

        try:
            with torch.inference_mode():
                output = model.generate(
                    **seg_inputs, max_new_tokens=int(max_new_tokens), do_sample=False
                )
            text = processor.tokenizer.decode(
                output[0, seg_inputs["input_ids"].shape[1]:], skip_special_tokens=True
            ).strip()
        except Exception as error:
            text = f"（生成失败：{type(error).__name__}: {error}）"
        log += f"[t={start:.1f}-{end:.1f}s] 🔊 响应 (p={probability:.2f})\n\n{text}\n\n"
        yield log


def clear_qa():
    return "", 0, 0.0, []


def clear_stream():
    return ""


def build_ui():
    """Build Gradio interface."""
    with gr.Blocks(title="Mage-VL Video Testing") as demo:
        gr.Markdown("# Mage-VL 视频测试\n上传视频、提问，并测试事件门控流式分析。")

        with gr.Tab("视频问答"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 模型配置")
                    # Prefer startup-preloaded paths, so offline servers don't fall
                    # back to a Hugging Face repo id they cannot reach.
                    model_choices = list(dict.fromkeys(_PRELOADED + ["microsoft/Mage-VL"]))
                    qa_model = gr.Dropdown(
                        choices=model_choices,
                        value=model_choices[0],
                        label="模型",
                    )
                    qa_custom = gr.Textbox(
                        label="自定义模型（可选）",
                        placeholder="Hugging Face repo id 或本地路径",
                        lines=1,
                    )

                    qa_backend = gr.Radio(
                        choices=["frames", "codec"],
                        value="frames",
                        label="视频后端",
                        info="frames: 均匀采样帧；codec: 编解码器处理",
                    )
                    qa_codec_engine = gr.Dropdown(
                        choices=["traditional", "neural"],
                        value="traditional",
                        label="编解码引擎（仅 codec 后端）",
                        info="traditional: HEVC/H.264；neural: DCVC-RT",
                        visible=False,
                    )

                    with gr.Row():
                        qa_num_frames = gr.Slider(
                            minimum=1, maximum=128, value=32, step=1,
                            label="帧数",
                            info="采样帧数或 codec 目标画布",
                        )
                        qa_max_pixels = gr.Slider(
                            minimum=50000, maximum=500000, value=150000, step=10000,
                            label="最大像素数（codec 后端）",
                        )

                    qa_max_new_tokens = gr.Slider(
                        minimum=64, maximum=1024, value=256, step=64,
                        label="最大生成 token 数",
                    )

                with gr.Column(scale=1):
                    gr.Markdown("### 视频输入")
                    qa_video = gr.Video(
                        label="上传视频",
                        sources=["upload"],
                    )
                    qa_question = gr.Textbox(
                        label="问题",
                        value="Describe this video.",
                        lines=2,
                    )

                    with gr.Row():
                        qa_run = gr.Button("分析视频", variant="primary", scale=2)
                        qa_clear = gr.Button("清除", scale=1)

            with gr.Column():
                gr.Markdown("### 结果")
                qa_answer = gr.Textbox(
                    label="模型回答（流式输出）",
                    lines=8,
                    interactive=False,
                )
                with gr.Row():
                    qa_tokens = gr.Number(label="Visual Tokens", precision=0)
                    qa_time = gr.Number(label="处理时间（秒）", precision=2)

                qa_frames = gr.Gallery(
                    label="采样帧（仅 frames 后端）",
                    columns=8,
                    rows=2,
                    height="auto",
                )

            gr.Markdown("### 示例")
            gr.Examples(
                examples=[
                    [os.path.join(os.path.dirname(__file__), "assets", "examples", "soccer-broadcast.mp4"),
                     "Describe this video."],
                ],
                inputs=[qa_video, qa_question],
            )

            qa_backend.change(
                lambda x: gr.update(visible=(x == "codec")),
                inputs=qa_backend,
                outputs=qa_codec_engine,
            )

            def run_qa(
                video_path, question, model_dropdown, custom_model,
                backend, codec_eng, num_frames_val, max_pixels_val, max_new_tokens_val,
                progress=gr.Progress(track_tqdm=False),
            ):
                model_path = (custom_model or "").strip() or model_dropdown
                yield from process_video(
                    video_path, question, model_path, backend, codec_eng,
                    num_frames_val, max_pixels_val, max_new_tokens_val, progress,
                )

            qa_run.click(
                fn=clear_qa,
                inputs=None,
                outputs=[qa_answer, qa_tokens, qa_time, qa_frames],
            ).then(
                fn=run_qa,
                inputs=[
                    qa_video, qa_question, qa_model, qa_custom, qa_backend,
                    qa_codec_engine, qa_num_frames, qa_max_pixels, qa_max_new_tokens,
                ],
                outputs=[qa_answer, qa_tokens, qa_time, qa_frames],
            )

            qa_clear.click(
                fn=clear_qa,
                inputs=None,
                outputs=[qa_answer, qa_tokens, qa_time, qa_frames],
            )

        with gr.Tab("门控流式分析"):
            gr.Markdown(
                "按固定时长切分视频，用 **StreamMind 认知门控**为每段打分："
                "分数低于阈值则保持沉默，达到阈值才调用 VLM 生成解说。"
                "门控在 codec 输入上训练，因此推荐使用 codec 后端。"
            )
            with gr.Row():
                with gr.Column(scale=1):
                    st_model = gr.Dropdown(
                        choices=list(dict.fromkeys(_PRELOADED + ["microsoft/Mage-VL"])),
                        value=list(dict.fromkeys(_PRELOADED + ["microsoft/Mage-VL"]))[0],
                        label="模型",
                    )
                    st_custom = gr.Textbox(
                        label="自定义模型（可选）",
                        placeholder="Hugging Face repo id 或本地路径",
                        lines=1,
                    )
                    st_backend = gr.Radio(
                        choices=["codec", "frames"],
                        value="codec",
                        label="视频后端",
                        info="门控在 codec 输入上训练，推荐 codec",
                    )
                    with gr.Row():
                        st_segment_sec = gr.Slider(
                            minimum=1.0, maximum=30.0, value=8.0, step=0.5,
                            label="片段时长（秒）",
                        )
                        st_num_frames = gr.Slider(
                            minimum=1, maximum=64, value=16, step=1,
                            label="每片段帧数（仅 frames 后端）",
                        )
                    with gr.Row():
                        st_cur_fps = gr.Slider(
                            minimum=0.5, maximum=8.0, value=2.0, step=0.5,
                            label="采样帧率（仅 frames 后端）",
                        )
                        st_threshold = gr.Slider(
                            minimum=0.0, maximum=1.0, value=0.5, step=0.05,
                            label="门控阈值",
                        )
                    with gr.Row():
                        st_max_new_tokens = gr.Slider(
                            minimum=16, maximum=512, value=80, step=16,
                            label="每段最大 token 数",
                        )
                        st_max_segments = gr.Slider(
                            minimum=0, maximum=64, value=0, step=1,
                            label="最多处理片段数（0 = 全部）",
                        )
                    st_prompt = gr.Textbox(
                        label="解说提示词",
                        value=USER_PROMPT,
                        lines=2,
                    )
                    st_run = gr.Button("开始流式分析", variant="primary")
                    st_clear = gr.Button("清除")

                with gr.Column(scale=1):
                    st_video = gr.Video(label="上传视频", sources=["upload"])
                    st_log = gr.Textbox(
                        label="门控决策与解说（流式输出）",
                        lines=22,
                        interactive=False,
                    )

            gr.Markdown("### 示例")
            gr.Examples(
                examples=[
                    [os.path.join(os.path.dirname(__file__), "assets", "examples", "soccer-broadcast.mp4")],
                ],
                inputs=[st_video],
            )

            def run_stream(
                video_path, model_dropdown, custom_model, backend, segment_sec,
                num_frames_val, cur_fps_val, threshold, max_new_tokens_val,
                prompt, max_segments_val, progress=gr.Progress(track_tqdm=False),
            ):
                model_path = (custom_model or "").strip() or model_dropdown
                yield from stream_video(
                    video_path, model_path, backend, segment_sec, num_frames_val,
                    cur_fps_val, threshold, max_new_tokens_val, prompt,
                    max_segments_val, progress,
                )

            st_run.click(
                fn=clear_stream, inputs=None, outputs=st_log,
            ).then(
                fn=run_stream,
                inputs=[
                    st_video, st_model, st_custom, st_backend, st_segment_sec,
                    st_num_frames, st_cur_fps, st_threshold, st_max_new_tokens,
                    st_prompt, st_max_segments,
                ],
                outputs=st_log,
            )

            st_clear.click(fn=clear_stream, inputs=None, outputs=st_log)

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
            model_path = model_path.strip()
            if model_path:
                load_model(model_path)
                _PRELOADED.append(model_path)

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
