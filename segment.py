"""使用本地 SAM 3 权重分割图片、目录或视频。"""
import argparse
import json
import time
from itertools import chain
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model

ROOT = Path(__file__).resolve().parent
CLASSES = [
    ("sky", (70, 155, 235)),
    ("person", (245, 150, 45)),
    ("car", (220, 75, 100)),
    ("electric scooter", (160, 100, 225)),
    ("bicycle", (45, 190, 130)),
]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi"}


def read_frames(path):
    """按需读取图片或视频帧，目录输入保留相对路径。"""
    if path.is_dir():
        for source in sorted(path.rglob("*")):
            if source.suffix.lower() in IMAGE_EXTENSIONS and source.is_file():
                with Image.open(source) as image:
                    yield source.relative_to(path), image.convert("RGB")
    elif path.suffix.lower() in VIDEO_EXTENSIONS:
        from decord import VideoReader

        for index, frame in enumerate(VideoReader(str(path))):
            yield Path(f"frame_{index:06d}"), Image.fromarray(frame.asnumpy())
    else:
        with Image.open(path) as image:
            yield Path(path.name), image.convert("RGB")


def save_prediction(processor, frame, destination, classes):
    """逐类推理，保存独立掩膜、标签及预览，返回统计信息。"""
    destination.mkdir(parents=True)
    overlay = frame.copy()
    labels = np.zeros((frame.height, frame.width), dtype=np.uint8)
    counts = {}
    with torch.autocast("cuda", dtype=torch.bfloat16):
        state = processor.set_image(frame)
        for class_id, (prompt, color) in enumerate(classes, start=1):
            processor.reset_all_prompts(state)
            result = processor.set_text_prompt(state=state, prompt=prompt)
            masks = result["masks"].flatten(0, 1)
            combined = masks.any(dim=0).cpu().numpy()
            counts[prompt] = {
                "instances": len(masks),
                "scores": result["scores"].float().cpu().tolist(),
                "pixels": int(combined.sum()),
            }
            mask = Image.fromarray(combined.astype(np.uint8) * 255)
            mask.save(destination / f"{class_id:02d}_mask.png")
            labels[combined] = class_id
            tint = Image.blend(overlay, Image.new("RGB", frame.size, color), 0.24)
            overlay = Image.composite(tint, overlay, mask)
            inner = np.asarray(mask.filter(ImageFilter.MinFilter(3)))
            edge = Image.fromarray((combined & (inner == 0)).astype(np.uint8) * 255)
            overlay.paste(color, mask=edge)
    Image.fromarray(labels).save(destination / "labels.png")
    overlay.save(destination / "overlay.jpg", quality=95)
    return counts


def parse_args():
    """验证输入、权重和输出路径，避免覆盖已有结果。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="图片、目录或视频")
    parser.add_argument("--output", type=Path, required=True, help="新的输出目录")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "weights/sam3.pt")
    parser.add_argument("--prompts", nargs="+", help="自定义类别")
    parser.add_argument("--threshold", type=float, default=0.35)
    args = parser.parse_args()
    if not args.input.exists() or not args.checkpoint.is_file():
        parser.error("请检查输入路径和本地权重路径")
    if args.output.exists():
        parser.error("输出目录已存在，请使用新的目录")
    if args.output.resolve().is_relative_to(args.input.resolve()):
        parser.error("输出目录请放在输入目录外")
    if not 0 <= args.threshold <= 1:
        parser.error("threshold 必须位于 0 到 1 之间")
    if not torch.cuda.is_available():
        parser.error("当前环境未检测到 CUDA GPU")
    prompts = args.prompts or [name for name, _ in CLASSES]
    if len(prompts) > 255 or len(set(prompts)) != len(prompts):
        parser.error("类别必须唯一且最多 255 个")
    return args, prompts


@torch.inference_mode()
def main():
    """运行离线 GPU 分割并记录加载和端到端处理耗时。"""
    args, prompts = parse_args()
    classes = [(prompt, CLASSES[i % len(CLASSES)][1]) for i, prompt in enumerate(prompts)]
    frames = read_frames(args.input)
    first = next(frames, None)
    if first is None:
        raise SystemExit("输入目录中没有支持的图片")
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    start = time.perf_counter()
    model = build_sam3_image_model(
        device="cuda", checkpoint_path=str(args.checkpoint), load_from_HF=False
    )
    processor = Sam3Processor(model, device="cuda", confidence_threshold=args.threshold)
    torch.cuda.synchronize()
    report = {
        "model_load_seconds": time.perf_counter() - start,
        "classes": {i: {"prompt": p, "color": c} for i, (p, c) in enumerate(classes, 1)},
        "images": {},
    }
    args.output.mkdir(parents=True)
    for name, frame in chain([first], frames):
        start = time.perf_counter()
        counts = save_prediction(processor, frame, args.output / name, classes)
        elapsed = time.perf_counter() - start
        report["images"][name.as_posix()] = {"seconds": elapsed, "classes": counts}
        (args.output / "summary.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"{name}: {elapsed:.3f}s", flush=True)


if __name__ == "__main__":
    main()
