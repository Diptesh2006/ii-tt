import os
import rasterio
import numpy as np
import cv2
from PIL import Image
import json

def get_thumbnail_path(tif_path, cache_dir):
    """Returns the path to the thumbnail for a given TIF."""
    base_name = os.path.splitext(os.path.basename(tif_path))[0]
    return os.path.join(cache_dir, f"{base_name}_thumb.png")

def generate_thumbnail(tif_path, output_path, max_dim=1024):
    """Generates a low-res PNG thumbnail from a large TIF."""
    if os.path.exists(output_path):
        return output_path
        
    with rasterio.open(tif_path) as src:
        # Calculate scale factor
        scale = max(src.width, src.height) / max_dim
        new_w = int(src.width / scale)
        new_h = int(src.height / scale)
        
        # Read downsampled RGB bands (1, 2, 3)
        data = src.read(
            [1, 2, 3],
            out_shape=(3, new_h, new_w),
            resampling=rasterio.enums.Resampling.bilinear
        )
        
        # Convert to HWC format for OpenCV/PIL
        img_hwc = data.transpose(1, 2, 0)
        
        # Robust min-max scaling for visualization (handle bit depths > 8)
        if img_hwc.max() > 255:
            # Simple linear scaling to 0-255
            img_hwc = ((img_hwc - img_hwc.min()) / (img_hwc.max() - img_hwc.min()) * 255).astype(np.uint8)
        else:
            img_hwc = img_hwc.astype(np.uint8)
            
        # Save as PNG
        Image.fromarray(img_hwc).save(output_path)
    
    return output_path

def save_analytics_cache(stats_path, data):
    """Saves analytics data to a JSON file."""
    with open(stats_path, 'w') as f:
        json.dump(data, f, indent=4)

def load_analytics_cache(stats_path):
    """Loads analytics data from a JSON file."""
    if os.path.exists(stats_path):
        with open(stats_path, 'r') as f:
            return json.load(f)
    return None

def get_cache_paths(tif_path, cache_dir):
    """Returns all relevant cache paths for a TIF."""
    base_name = os.path.splitext(os.path.basename(tif_path))[0]
    return {
        "mask": os.path.join(cache_dir, f"{base_name}_mask.tif"),
        "overlay": os.path.join(cache_dir, f"{base_name}_overlay.tif"),
        "stats": os.path.join(cache_dir, f"{base_name}_stats.json"),
        "thumbnail": os.path.join(cache_dir, f"{base_name}_thumb.png")
    }
