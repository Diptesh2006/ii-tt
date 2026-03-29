import os
import torch
import numpy as np
import rasterio
from rasterio.windows import Window
from PIL import Image
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
import torch.nn.functional as F
import cv2
import time

# Constants
CLASS_NAMES = ["Background", "Road", "Built-up", "Water"]
NUM_CLASSES = 4
CLASS_COLORS = {
    0: (60, 60, 60),   # Background
    1: (240, 180, 25),   # Road - Amber
    2: (215, 50, 50),    # Built-up - Red
    3: (25, 115, 215),     # Water - Blue
}

def load_model(model_path, device):
    """Loads the SegFormer model and processor."""
    # Handle nested directory structure if needed
    actual_model_path = model_path
    if not os.path.exists(os.path.join(actual_model_path, "config.json")):
        sub_dir = os.path.join(actual_model_path, os.path.basename(actual_model_path))
        if os.path.exists(os.path.join(sub_dir, "config.json")):
            actual_model_path = sub_dir
            
    processor = SegformerImageProcessor.from_pretrained(actual_model_path)
    model = SegformerForSemanticSegmentation.from_pretrained(
        actual_model_path, 
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True
    ).to(device)
    model.eval()
    return model, processor

def apply_color_map(mask, colors):
    """Converts a (H, W) label mask to (3, H, W) RGB image."""
    h, w = mask.shape
    rgb = np.zeros((3, h, w), dtype=np.uint8)
    for label, color in colors.items():
        mask_indices = (mask == label)
        for i in range(3): # R, G, B
            rgb[i][mask_indices] = color[i]
    return rgb

def run_tiled_inference(image_path, model, processor, device, output_path, 
                        mode='overlay', alpha=0.5, crop_size=1024, 
                        tile_size=512, overlap=128, progress_callback=None):
    """
    Runs inference on a large TIF file using a sliding window approach.
    Reports progress via progress_callback(current, total).
    Returns class_counts, transform, crs, total_time, avg_tile_time
    """
    start_time_total = time.time()
    tile_times = []
    
    with rasterio.open(image_path) as src:
        profile = src.profile.copy()
        h, w = src.height, src.width
        
        # Update profile for output
        if mode == 'mask':
            profile.update(dtype=rasterio.uint8, count=1, nodata=255)
        else:
            profile.update(dtype=rasterio.uint8, count=3, nodata=0)
        
        stride = crop_size - overlap
        n_tiles_y = (h + stride - 1) // stride
        n_tiles_x = (w + stride - 1) // stride
        total_tiles = n_tiles_y * n_tiles_x
        current_tile = 0
        
        # Track class counts for analytics
        class_counts = np.zeros(NUM_CLASSES, dtype=np.int64)
        
        with rasterio.open(output_path, 'w', **profile) as dst:
            for y in range(0, h, stride):
                for x in range(0, w, stride):
                    tile_start_time = time.time()
                    window_h = min(crop_size, h - y)
                    window_w = min(crop_size, w - x)
                    window = Window(x, y, window_w, window_h)
                    
                    # Read image data (3 bands)
                    img_data = src.read([1, 2, 3], window=window)
                    
                    # Valid mask for padding
                    valid_mask = src.read_masks(1, window=window)
                    is_black = np.all(img_data == 0, axis=0)
                    final_valid_mask = (valid_mask > 0) & (~is_black)
                    
                    # Pad
                    if window_h < crop_size or window_w < crop_size:
                        padded_img = np.zeros((3, crop_size, crop_size), dtype=img_data.dtype)
                        padded_img[:, :window_h, :window_w] = img_data
                    else:
                        padded_img = img_data
                        
                    # Resize for model
                    img_hwc = padded_img.transpose(1, 2, 0)
                    resized_hwc = cv2.resize(img_hwc, (tile_size, tile_size), interpolation=cv2.INTER_LINEAR)
                    
                    # Infer
                    img_pil = Image.fromarray(resized_hwc)
                    inputs = processor(images=img_pil, return_tensors="pt").to(device)
                    
                    with torch.no_grad():
                        outputs = model(**inputs)
                        logits = F.interpolate(outputs.logits, size=(crop_size, crop_size), mode="bilinear", align_corners=False)
                        labels = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
                    
                    # Crop back
                    final_labels = labels[:window_h, :window_w]
                    
                    # Update stats (only for valid pixels)
                    if final_valid_mask.any():
                        valid_pixels = final_labels[final_valid_mask]
                        for c in range(NUM_CLASSES):
                            class_counts[c] += np.sum(valid_pixels == c)
                    
                    # Mask invalid
                    final_labels[~final_valid_mask] = 255
                    
                    # Write output
                    if mode == 'mask':
                        dst.write(final_labels, 1, window=window)
                    elif mode == 'color':
                        color_tile = apply_color_map(final_labels, CLASS_COLORS)
                        dst.write(color_tile, window=window)
                    elif mode == 'overlay':
                        color_tile = apply_color_map(final_labels, CLASS_COLORS)
                        original = img_data[:, :window_h, :window_w]
                        blended = (alpha * color_tile + (1 - alpha) * original).astype(np.uint8)
                        
                        # Background (class 3) transparency
                        bg_mask = (final_labels == 3)
                        for c in range(3):
                            blended[c, bg_mask] = original[c, bg_mask]
                            
                        # Keep invalid black
                        blended[:, ~final_valid_mask] = 0
                        dst.write(blended, window=window)
                    
                    tile_end_time = time.time()
                    tile_times.append(tile_end_time - tile_start_time)
                    
                    current_tile += 1
                    if progress_callback:
                        progress_callback(current_tile, total_tiles)
        
        total_time = time.time() - start_time_total
        avg_tile_time = sum(tile_times) / len(tile_times) if tile_times else 0.0
        
        return class_counts, profile.get('transform'), profile.get('crs'), total_time, avg_tile_time

