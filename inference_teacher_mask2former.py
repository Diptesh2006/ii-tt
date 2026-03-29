#!/usr/bin/env python3
"""
Standalone inference for the teacher Mask2Former checkpoint saved in `teacher_mask2former`.

Examples
--------
python inference_teacher_mask2former.py --input image.png
python inference_teacher_mask2former.py --input tile.npy --mode color
python inference_teacher_mask2former.py --input ortho.tif --mode overlay --output ortho_teacher_overlay.tif
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from PIL import Image


torch = None
F = None
tqdm = None
rasterio = None
Window = None
Mask2FormerForUniversalSegmentation = None
Mask2FormerImageProcessor = None


DEFAULT_MODEL_DIR = "teacher_mask2former"
BACKGROUND_LABEL = 0

DEFAULT_ID2LABEL = {
    0: "Background",
    1: "Road",
    2: "Water",
    3: "Built-up",
    4: "Bridge",
    5: "Railway",
}

CLASS_COLORS = {
    0: (60, 60, 60),      # Background
    1: (240, 180, 25),    # Road
    2: (25, 115, 215),    # Water
    3: (215, 50, 50),     # Built-up
    4: (156, 102, 31),    # Bridge
    5: (46, 153, 67),     # Railway
}

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".npy"}
GEOTIFF_EXTENSIONS = {".tif", ".tiff"}
PIL_BILINEAR = Image.Resampling.BILINEAR if hasattr(Image, "Resampling") else Image.BILINEAR


def load_runtime_dependencies(require_rasterio: bool) -> None:
    global torch, F, tqdm, rasterio, Window
    global Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor

    if torch is None:
        import torch as _torch
        import torch.nn.functional as _F
        from tqdm import tqdm as _tqdm
        from transformers import (
            Mask2FormerForUniversalSegmentation as _Mask2FormerForUniversalSegmentation,
            Mask2FormerImageProcessor as _Mask2FormerImageProcessor,
        )

        torch = _torch
        F = _F
        tqdm = _tqdm
        Mask2FormerForUniversalSegmentation = _Mask2FormerForUniversalSegmentation
        Mask2FormerImageProcessor = _Mask2FormerImageProcessor

    if require_rasterio and rasterio is None:
        try:
            import rasterio as _rasterio
            from rasterio.windows import Window as _Window
        except ImportError as exc:  # pragma: no cover - depends on local environment
            raise ImportError("GeoTIFF inference requires 'rasterio'. Install it first, then rerun.") from exc

        rasterio = _rasterio
        Window = _Window


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run standalone inference with the local teacher Mask2Former checkpoint."
    )
    parser.add_argument("--input", required=True, help="Input image path (.png/.jpg/.npy/.tif/.tiff).")
    parser.add_argument(
        "--model-dir",
        default=DEFAULT_MODEL_DIR,
        help=f"Checkpoint directory. Defaults to ./{DEFAULT_MODEL_DIR}",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path. Defaults to '<input>_teacher_<mode>.png' or '.tif' for GeoTIFFs.",
    )
    parser.add_argument(
        "--mode",
        choices=("mask", "color", "overlay"),
        default="mask",
        help="mask: class-id map, color: colorized prediction, overlay: blended on original RGB.",
    )
    parser.add_argument("--tile-size", type=int, default=512, help="Model input size used for resizing.")
    parser.add_argument("--crop-size", type=int, default=1024, help="Window size for tiled GeoTIFF inference.")
    parser.add_argument("--overlap", type=int, default=128, help="Window overlap for tiled GeoTIFF inference.")
    parser.add_argument("--alpha", type=float, default=0.5, help="Overlay blend ratio.")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Inference device selection.",
    )
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            print("CUDA requested but not available. Falling back to CPU.")
            return torch.device("cpu")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_model_dir(model_dir: str) -> Path:
    root = Path(model_dir).expanduser().resolve()
    if (root / "config.json").exists():
        return root

    nested = root / root.name
    if (nested / "config.json").exists():
        return nested

    raise FileNotFoundError(f"Could not find a Hugging Face checkpoint under: {model_dir}")


def resolve_output_path(input_path: Path, output_path: str | None, is_geotiff: bool, mode: str) -> Path:
    if output_path:
        resolved = Path(output_path).expanduser().resolve()
    else:
        suffix = ".tif" if is_geotiff else ".png"
        resolved = input_path.resolve().with_name(f"{input_path.stem}_teacher_{mode}{suffix}")

    if is_geotiff and resolved.suffix.lower() not in GEOTIFF_EXTENSIONS:
        raise ValueError("GeoTIFF output must end with .tif or .tiff.")

    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def get_id2label(model: Mask2FormerForUniversalSegmentation) -> Dict[int, str]:
    raw = getattr(model.config, "id2label", None) or {}
    id2label = {int(key): value for key, value in raw.items()}
    if not id2label:
        return DEFAULT_ID2LABEL.copy()

    generic = all(name.startswith("LABEL_") for name in id2label.values())
    if generic and len(id2label) == len(DEFAULT_ID2LABEL):
        return DEFAULT_ID2LABEL.copy()

    return id2label


def load_model(model_dir: str, device: torch.device):
    resolved_dir = resolve_model_dir(model_dir)
    processor = Mask2FormerImageProcessor.from_pretrained(str(resolved_dir))
    model = Mask2FormerForUniversalSegmentation.from_pretrained(str(resolved_dir)).to(device)
    model.eval()

    mean = np.asarray(processor.image_mean, dtype=np.float32).reshape(1, 1, 3)
    std = np.asarray(processor.image_std, dtype=np.float32).reshape(1, 1, 3)
    id2label = get_id2label(model)
    return model, mean, std, id2label, resolved_dir


def ensure_hwc_rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)

    if image.ndim == 3 and image.shape[0] in (1, 3, 4) and image.shape[1] > 8 and image.shape[2] > 8:
        image = np.transpose(image, (1, 2, 0))

    if image.ndim != 3:
        raise ValueError(f"Expected a 2D or 3D image array, got shape {image.shape}.")

    if image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)
    elif image.shape[2] == 2:
        image = np.concatenate([image, image[..., 1:2]], axis=2)
    elif image.shape[2] >= 3:
        image = image[..., :3]
    else:
        raise ValueError(f"Unsupported channel layout: {image.shape}")

    return image


def to_uint8_rgb(image: np.ndarray) -> np.ndarray:
    image = ensure_hwc_rgb(np.asarray(image))
    original_dtype = image.dtype
    image = image.astype(np.float32)

    if np.issubdtype(original_dtype, np.integer):
        info = np.iinfo(original_dtype)
        if info.max > 255:
            image = image / float(info.max) * 255.0
    elif image.max() <= 1.0 + 1e-6:
        image = image * 255.0
    elif image.max() > 255.0:
        min_val = float(image.min())
        max_val = float(image.max())
        if max_val > min_val:
            image = (image - min_val) / (max_val - min_val) * 255.0
        else:
            image = np.zeros_like(image)

    return np.clip(image, 0, 255).astype(np.uint8)


def preprocess_image(image_uint8: np.ndarray, mean: np.ndarray, std: np.ndarray, tile_size: int) -> torch.Tensor:
    resized = np.asarray(
        Image.fromarray(image_uint8).resize((tile_size, tile_size), resample=PIL_BILINEAR),
        dtype=np.float32,
    )
    normalized = resized / 255.0
    normalized = (normalized - mean) / std
    chw = np.transpose(normalized, (2, 0, 1))
    return torch.from_numpy(chw).unsqueeze(0)


def mask2former_semantic_logits(outputs, target_hw: Tuple[int, int]) -> torch.Tensor:
    # Teacher training used the same query-to-semantic conversion.
    class_probs = F.softmax(outputs.class_queries_logits, dim=-1)[..., :-1]
    mask_probs = outputs.masks_queries_logits.sigmoid()
    semantic = torch.einsum("bqc,bqhw->bchw", class_probs, mask_probs)
    return F.interpolate(semantic, size=target_hw, mode="bilinear", align_corners=False)


def predict_mask(
    image_uint8: np.ndarray,
    model: Mask2FormerForUniversalSegmentation,
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
    tile_size: int,
    target_hw: Tuple[int, int] | None = None,
) -> np.ndarray:
    if target_hw is None:
        target_hw = image_uint8.shape[:2]

    pixel_values = preprocess_image(image_uint8, mean, std, tile_size).to(device)

    with torch.inference_mode():
        outputs = model(pixel_values=pixel_values)
        semantic = mask2former_semantic_logits(outputs, target_hw)
        labels = semantic.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    return labels


def apply_color_map(mask: np.ndarray) -> np.ndarray:
    color = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for label, rgb in CLASS_COLORS.items():
        color[mask == label] = rgb
    return color


def build_output_image(
    rgb_uint8: np.ndarray,
    labels: np.ndarray,
    mode: str,
    alpha: float,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray | np.uint8:
    if mode == "mask":
        return labels

    color = apply_color_map(labels)
    if valid_mask is not None:
        color[~valid_mask] = 0

    if mode == "color":
        return color

    blended = (
        alpha * color.astype(np.float32)
        + (1.0 - alpha) * rgb_uint8.astype(np.float32)
    ).astype(np.uint8)
    blended[labels == BACKGROUND_LABEL] = rgb_uint8[labels == BACKGROUND_LABEL]
    if valid_mask is not None:
        blended[~valid_mask] = 0
    return blended


def print_class_counts(class_counts: np.ndarray, id2label: Dict[int, str]) -> None:
    print("Predicted pixel counts:")
    for class_id in range(len(class_counts)):
        label = id2label.get(class_id, f"Class {class_id}")
        print(f"  {class_id}: {label:<10} {int(class_counts[class_id])}")


def load_regular_image(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        image = np.load(path)
        return to_uint8_rgb(image)

    with Image.open(path) as img:
        return np.asarray(img.convert("RGB"), dtype=np.uint8)


def run_single_image_inference(
    input_path: Path,
    output_path: Path,
    model: Mask2FormerForUniversalSegmentation,
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
    tile_size: int,
    mode: str,
    alpha: float,
    id2label: Dict[int, str],
) -> None:
    rgb_uint8 = load_regular_image(input_path)
    labels = predict_mask(rgb_uint8, model, mean, std, device, tile_size, rgb_uint8.shape[:2])

    output = build_output_image(rgb_uint8, labels, mode, alpha)
    if mode == "mask":
        Image.fromarray(output).save(output_path)
    else:
        Image.fromarray(output).save(output_path)

    class_counts = np.bincount(labels.reshape(-1), minlength=len(id2label))
    print_class_counts(class_counts, id2label)
    print(f"Saved output to: {output_path}")


def read_raster_rgb(src: rasterio.io.DatasetReader, window: Window) -> np.ndarray:
    band_indexes = list(range(1, min(src.count, 3) + 1))
    data = src.read(band_indexes, window=window)
    return to_uint8_rgb(data)


def run_tiled_geotiff_inference(
    input_path: Path,
    output_path: Path,
    model: Mask2FormerForUniversalSegmentation,
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
    tile_size: int,
    crop_size: int,
    overlap: int,
    mode: str,
    alpha: float,
    id2label: Dict[int, str],
) -> None:
    stride = crop_size - overlap
    if stride <= 0:
        raise ValueError("overlap must be smaller than crop-size.")

    class_counts = np.zeros(len(id2label), dtype=np.int64)

    with rasterio.open(input_path) as src:
        profile = src.profile.copy()
        profile.update(dtype=rasterio.uint8)
        if mode == "mask":
            profile.update(count=1, nodata=255)
        else:
            profile.update(count=3, nodata=0)

        height, width = src.height, src.width
        total_tiles_y = (height + stride - 1) // stride
        total_tiles_x = (width + stride - 1) // stride
        total_tiles = total_tiles_y * total_tiles_x

        with rasterio.open(output_path, "w", **profile) as dst:
            progress = tqdm(total=total_tiles, desc="Teacher Mask2Former inference")

            for top in range(0, height, stride):
                for left in range(0, width, stride):
                    window_h = min(crop_size, height - top)
                    window_w = min(crop_size, width - left)
                    window = Window(left, top, window_w, window_h)

                    rgb_uint8 = read_raster_rgb(src, window)
                    valid_mask = src.read_masks(1, window=window) > 0
                    is_black = np.all(rgb_uint8 == 0, axis=2)
                    final_valid_mask = valid_mask & (~is_black)

                    padded_rgb = np.zeros((crop_size, crop_size, 3), dtype=np.uint8)
                    padded_rgb[:window_h, :window_w] = rgb_uint8

                    labels = predict_mask(
                        padded_rgb,
                        model,
                        mean,
                        std,
                        device,
                        tile_size,
                        target_hw=(crop_size, crop_size),
                    )
                    labels = labels[:window_h, :window_w]

                    valid_pixels = labels[final_valid_mask]
                    if valid_pixels.size:
                        class_counts += np.bincount(valid_pixels, minlength=len(id2label))

                    labels[~final_valid_mask] = 255

                    if mode == "mask":
                        dst.write(labels, 1, window=window)
                    else:
                        output = build_output_image(rgb_uint8, labels, mode, alpha, final_valid_mask)
                        dst.write(np.transpose(output, (2, 0, 1)), window=window)

                    progress.update(1)

            progress.close()

    print_class_counts(class_counts, id2label)
    print(f"Saved output to: {output_path}")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if input_path.suffix.lower() not in IMAGE_EXTENSIONS:
        raise ValueError(f"Unsupported input extension: {input_path.suffix}")

    is_geotiff = input_path.suffix.lower() in GEOTIFF_EXTENSIONS
    load_runtime_dependencies(require_rasterio=is_geotiff)
    output_path = resolve_output_path(input_path, args.output, is_geotiff, args.mode)
    device = resolve_device(args.device)

    model, mean, std, id2label, resolved_model_dir = load_model(args.model_dir, device)

    print(f"Model     : {resolved_model_dir}")
    print(f"Input     : {input_path}")
    print(f"Output    : {output_path}")
    print(f"Device    : {device}")
    print(f"Classes   : {[id2label[i] for i in sorted(id2label)]}")

    if is_geotiff:
        run_tiled_geotiff_inference(
            input_path=input_path,
            output_path=output_path,
            model=model,
            mean=mean,
            std=std,
            device=device,
            tile_size=args.tile_size,
            crop_size=args.crop_size,
            overlap=args.overlap,
            mode=args.mode,
            alpha=args.alpha,
            id2label=id2label,
        )
    else:
        run_single_image_inference(
            input_path=input_path,
            output_path=output_path,
            model=model,
            mean=mean,
            std=std,
            device=device,
            tile_size=args.tile_size,
            mode=args.mode,
            alpha=args.alpha,
            id2label=id2label,
        )


if __name__ == "__main__":
    main()
