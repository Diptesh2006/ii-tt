import os
from pathlib import Path

import numpy as np
import geopandas as gpd
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.windows import Window
from shapely.geometry import box
from tqdm import tqdm
import cv2

TILE_SIZE = 512
CROP_SIZE = 1024  # Increased for more zoom out as requested
STRIDE = 768  # Adjusted for larger crop size
MIN_FOREGROUND_RATIO = 0.01
CLASS_BG = 0
CLASS_ROAD = 1
CLASS_BUILT = 2
CLASS_WATER = 3
CLASS_BRIDGE = 4
CLASS_RAILWAY = 5

ROAD_KEYS = ["road"]
BUILT_KEYS = ["built"]
WATER_KEYS = ["water"]
BRIDGE_KEYS = ["bridge"]
RAILWAY_KEYS = ["railway"]

# ============ DATASET MAPPINGS ============

DATASETS = [
    {
        "tif_dir": Path(r"D:\geo-dataste\CG\tifs"),
        "shp_dir": r"D:\geo-dataste\CG\shp-file",
        "out_dir": r"D:\geo-dataste\zoomed_out_1024_updated\train",
        "prefix": "CG"
    },
    {
        "tif_dir": Path(r"D:\geo-dataste\PB_training_dataSet_shp_file\tifs"),
        "shp_dir": r"D:\geo-dataste\PB_training_dataSet_shp_file\shp-file",
        "out_dir": r"D:\geo-dataste\zoomed_out_1024_updated\train",
        "prefix": "PB"
    }
]

# =========================================


def find_shapefiles(shp_dir):
    shp_dir = Path(shp_dir)
    shp_map = {"road": [], "built": [], "water": [], "bridge": [], "railway": []}

    for shp in shp_dir.glob("*.shp"):
        name = shp.name.lower()
        if any(k in name for k in ROAD_KEYS):
            shp_map["road"].append(shp)
        elif any(k in name for k in BUILT_KEYS):
            shp_map["built"].append(shp)
        elif any(k in name for k in WATER_KEYS):
            shp_map["water"].append(shp)
        elif any(k in name for k in BRIDGE_KEYS):
            shp_map["bridge"].append(shp)
        elif any(k in name for k in RAILWAY_KEYS):
            shp_map["railway"].append(shp)

    return shp_map


def load_and_clip(shp_list, crs, bounds):
    if not shp_list:
        return gpd.GeoDataFrame(geometry=[], crs=crs)

    bbox = box(bounds.left, bounds.bottom, bounds.right, bounds.top)
    frames = []

    for shp in shp_list:
        gdf = gpd.read_file(shp).to_crs(crs)

        gdf["geometry"] = gdf.geometry.buffer(0)
        gdf = gdf[gdf.is_valid]
        gdf = gdf[gdf.intersects(bbox)]

        frames.append(gdf)

    if not frames:
        return gpd.GeoDataFrame(geometry=[], crs=crs)

    return gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=crs)


def get_geometries(tif_path, shp_map):
    with rasterio.open(tif_path) as src:
        bounds = src.bounds
        crs = src.crs

    geoms_map = {
        "road": load_and_clip(shp_map["road"], crs, bounds),
        "built": load_and_clip(shp_map["built"], crs, bounds),
        "water": load_and_clip(shp_map["water"], crs, bounds),
        "bridge": load_and_clip(shp_map["bridge"], crs, bounds),
        "railway": load_and_clip(shp_map["railway"], crs, bounds)
    }
    return geoms_map


def tile_and_save(tif_path, geoms_map, out_dir, prefix):
    out_img = Path(out_dir) / "images"
    out_mask = Path(out_dir) / "masks"
    out_img.mkdir(parents=True, exist_ok=True)
    out_mask.mkdir(parents=True, exist_ok=True)

    with rasterio.open(tif_path) as src:
        H, W = src.height, src.width
        transform = src.transform

        road = geoms_map["road"]
        built = geoms_map["built"]
        water = geoms_map["water"]
        bridge = geoms_map["bridge"]
        railway = geoms_map["railway"]

        tid = 0
        for y in tqdm(range(0, H - CROP_SIZE + 1, STRIDE)):
            for x in range(0, W - CROP_SIZE + 1, STRIDE):

                window = Window(x, y, CROP_SIZE, CROP_SIZE)
                win_transform = src.window_transform(window)
                win_bounds = src.window_bounds(window)
                win_box = box(*win_bounds)

                # Rasterize tile mask
                mask_t = np.zeros((CROP_SIZE, CROP_SIZE), dtype=np.uint8)

                # Water
                if not water.empty:
                    hit = water[water.intersects(win_box)]
                    if not hit.empty:
                        rasterize(
                            [(g, CLASS_WATER) for g in hit.geometry],
                            out_shape=(CROP_SIZE, CROP_SIZE),
                            transform=win_transform,
                            out=mask_t
                        )

                # Built
                if not built.empty:
                    hit = built[built.intersects(win_box)]
                    if not hit.empty:
                        rasterize(
                            [(g, CLASS_BUILT) for g in hit.geometry],
                            out_shape=(CROP_SIZE, CROP_SIZE),
                            transform=win_transform,
                            out=mask_t
                        )

                # Bridge
                if not bridge.empty:
                    hit = bridge[bridge.intersects(win_box)]
                    if not hit.empty:
                        rasterize(
                            [(g, CLASS_BRIDGE) for g in hit.geometry],
                            out_shape=(CROP_SIZE, CROP_SIZE),
                            transform=win_transform,
                            out=mask_t
                        )

                # Railway
                if not railway.empty:
                    hit = railway[railway.intersects(win_box)]
                    if not hit.empty:
                        rasterize(
                            [(g, CLASS_RAILWAY) for g in hit.geometry],
                            out_shape=(CROP_SIZE, CROP_SIZE),
                            transform=win_transform,
                            out=mask_t
                        )

                # Road
                if not road.empty:
                    hit = road[road.intersects(win_box)]
                    if not hit.empty:
                        rasterize(
                            [(g, CLASS_ROAD) for g in hit.geometry],
                            out_shape=(CROP_SIZE, CROP_SIZE),
                            transform=win_transform,
                            all_touched=True,
                            out=mask_t
                        )

                # Resize mask to TILE_SIZE
                mask_t = cv2.resize(mask_t, (TILE_SIZE, TILE_SIZE), interpolation=cv2.INTER_NEAREST)

                fg_ratio = (mask_t > 0).sum() / (TILE_SIZE * TILE_SIZE)
                if fg_ratio < MIN_FOREGROUND_RATIO:
                    continue

                img_t = src.read(window=window)
                
                # Check if image has 4 channels, if so, we might want to keep or discard Alpha
                # The user asked about channels, all have 4 (RGBA). Usually we take RGB.
                if img_t.shape[0] == 4:
                    img_t = img_t[:3, ...]

                # Resize image to TILE_SIZE
                img_t = cv2.resize(img_t.transpose(1, 2, 0), (TILE_SIZE, TILE_SIZE), interpolation=cv2.INTER_LINEAR).transpose(2, 0, 1)

                np.save(out_img / f"{prefix}_{tid:06d}.npy", img_t)
                np.save(out_mask / f"{prefix}_{tid:06d}.npy", mask_t)

                tid += 1


def build_dataset(tif_path, shp_dir, out_dir, prefix):
    shp_map = find_shapefiles(shp_dir)
    geoms_map = get_geometries(tif_path, shp_map)
    tile_and_save(tif_path, geoms_map, out_dir, prefix)


# ================= RUN ====================

if __name__ == "__main__":

    for ds in DATASETS:
        tifs = sorted(ds["tif_dir"].glob("*.tif"))

        for idx, tif_path in enumerate(tifs, start=1):
            prefix = f"{ds['prefix']}{idx:03d}"
            print(f"\nProcessing {tif_path.name} -> prefix {prefix}")

            build_dataset(
                tif_path=str(tif_path),
                shp_dir=ds["shp_dir"],
                out_dir=ds["out_dir"],
                prefix=prefix
            )