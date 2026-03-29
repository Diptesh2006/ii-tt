import os
import argparse
import torch
import numpy as np
import rasterio
from rasterio.windows import Window
from tqdm import tqdm
from PIL import Image
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
import torch.nn.functional as F
import cv2

# ─────────────────────────────────────────────────
# CONFIG / CONSTANTS
# ─────────────────────────────────────────────────

# Default class names from testing_results.py
# 0: Water, 1: Road, 2: Built-up, 3: Background
CLASS_NAMES = ["Water", "Road", "Built-up", "Background"]
NUM_CLASSES = 4

# Colors for visualization (RGB)
CLASS_COLORS = {
    0: (25, 115, 215),   # Water - Blue
    1: (240, 180, 25),   # Road - Amber
    2: (215, 50, 50),    # Built-up - Red
    3: (60, 60, 60),     # Background - Dark Grey
}

# ─────────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────────

def load_model(model_path, device):
    print(f"Loading model and processor from: {model_path}")
    processor = SegformerImageProcessor.from_pretrained(model_path)
    model = SegformerForSemanticSegmentation.from_pretrained(
        model_path, 
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True
    ).to(device)
    model.eval()
    return model, processor

# ─────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────

def apply_color_map(mask, colors):
    """Converts a (H, W) label mask to (3, H, W) RGB image."""
    h, w = mask.shape
    rgb = np.zeros((3, h, w), dtype=np.uint8)
    for label, color in colors.items():
        for i in range(3): # R, G, B
            rgb[i][mask == label] = color[i]
    return rgb

# ─────────────────────────────────────────────────
# INFERENCE LOGIC
# ─────────────────────────────────────────────────

def run_inference(image_path, model, processor, device, output_path, mode='mask', alpha=0.5, crop_size=1024, tile_size=512, overlap=128):
    with rasterio.open(image_path) as src:
        profile = src.profile.copy()
        h, w = src.height, src.width
        
        # Update profile for output
        if mode == 'mask':
            profile.update(dtype=rasterio.uint8, count=1, nodata=255)
        else:
            # Color or Overlay mode -> 3 bands (RGB)
            profile.update(dtype=rasterio.uint8, count=3, nodata=0)
        
        stride = crop_size - overlap
        
        with rasterio.open(output_path, 'w', **profile) as dst:
            n_tiles_y = (h + stride - 1) // stride
            n_tiles_x = (w + stride - 1) // stride
            pbar = tqdm(total=n_tiles_y * n_tiles_x)
            
            for y in range(0, h, stride):
                for x in range(0, w, stride):
                    window_h = min(crop_size, h - y)
                    window_w = min(crop_size, w - x)
                    window = Window(x, y, window_w, window_h)
                    
                    # Read image data (must be 3 bands for the model)
                    img_data = src.read([1, 2, 3], window=window)
                    
                    # Create a valid mask to ignore outer padding
                    valid_mask = src.read_masks(1, window=window)
                    is_black = np.all(img_data == 0, axis=0)
                    final_valid_mask = (valid_mask > 0) & (~is_black)
                    
                    # Pad if smaller than crop_size
                    if window_h < crop_size or window_w < crop_size:
                        padded_img = np.zeros((3, crop_size, crop_size), dtype=img_data.dtype)
                        padded_img[:, :window_h, :window_w] = img_data
                        # Optional: reflection padding could be better but zero padding is safer here
                    else:
                        padded_img = img_data
                        
                    # Resize to tile_size (e.g., 512x512) for the model to match training scale
                    img_hwc = padded_img.transpose(1, 2, 0)
                    resized_hwc = cv2.resize(img_hwc, (tile_size, tile_size), interpolation=cv2.INTER_LINEAR)
                    
                    # Preprocess and Infer
                    img_pil = Image.fromarray(resized_hwc)
                    inputs = processor(images=img_pil, return_tensors="pt").to(device)
                    
                    with torch.no_grad():
                        outputs = model(**inputs)
                        # Interpolate outputs back up to the original extraction scale (crop_size)
                        logits = F.interpolate(outputs.logits, size=(crop_size, crop_size), mode="bilinear", align_corners=False)
                        labels = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
                    
                    # Crop result back to window size
                    final_labels = labels[:window_h, :window_w]
                    
                    # Mask out invalid pixels (nodata / pure black padding)
                    final_labels[~final_valid_mask] = 255
                    
                    # Determine what to write
                    if mode == 'mask':
                        dst.write(final_labels, 1, window=window)
                    elif mode == 'color':
                        color_tile = apply_color_map(final_labels, CLASS_COLORS)
                        dst.write(color_tile, window=window)
                    elif mode == 'overlay':
                        color_tile = apply_color_map(final_labels, CLASS_COLORS)
                        # Blend: alpha * color + (1-alpha) * original
                        # Ensure original is uint8 and handles shapes
                        original = img_data[:, :window_h, :window_w]
                        blended = (alpha * color_tile + (1 - alpha) * original).astype(np.uint8)
                        
                        # Make Background (class 3) fully transparent by restoring original pixels
                        bg_mask = (final_labels == 3)
                        for c in range(3):
                            blended[c, bg_mask] = original[c, bg_mask]
                            
                        # Ensure invalid areas remain black/transparent
                        blended[:, ~final_valid_mask] = 0
                        dst.write(blended, window=window)
                    
                    pbar.update(1)
            pbar.close()

# ─────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Segformer inference on a .tif file")
    parser.add_argument("--input", type=str, required=True, help="Path to input .tif file")
    parser.add_argument("--model", type=str, required=True, help="Path to model directory")
    parser.add_argument("--output", type=str, default="output_mask.tif", help="Path to output .tif file")
    parser.add_argument("--mode", type=str, choices=['mask', 'color', 'overlay'], default='mask', 
                        help="Output mode: 'mask' (grayscale), 'color' (RGB map), or 'overlay' (blended with original)")
    parser.add_argument("--alpha", type=float, default=0.5, help="Alpha transparency for overlay (0.0 to 1.0)")
    parser.add_argument("--crop_size", type=int, default=1024, help="Crop size extracted from TIF")
    parser.add_argument("--tile_size", type=int, default=512, help="Tile size for inference (model input)")
    parser.add_argument("--overlap", type=int, default=128, help="Overlap between crops")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Handle model directory
    actual_model_path = args.model
    if not os.path.exists(os.path.join(actual_model_path, "config.json")):
        sub_dir = os.path.join(actual_model_path, os.path.basename(actual_model_path))
        if os.path.exists(os.path.join(sub_dir, "config.json")):
            actual_model_path = sub_dir
    
    model, processor = load_model(actual_model_path, device)
    
    print(f"Starting inference on {args.input} (mode: {args.mode})...")
    run_inference(
        args.input, 
        model, 
        processor, 
        device, 
        args.output, 
        mode=args.mode,
        alpha=args.alpha,
        crop_size=args.crop_size,
        tile_size=args.tile_size, 
        overlap=args.overlap
    )
    print(f"Finished! Output saved to {args.output}")
