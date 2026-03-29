import rasterio
import numpy as np
import os
import subprocess

def create_dummy_tif(path, width=1024, height=1024):
    """Creates a dummy 3-band RGB GeoTIFF."""
    data = np.random.randint(0, 256, (3, height, width), dtype=np.uint8)
    
    with rasterio.open(
        path,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=3,
        dtype='uint8',
        crs='EPSG:4326',
        transform=rasterio.transform.from_origin(0, 0, 0.0001, 0.0001)
    ) as dst:
        dst.write(data)
    print(f"Created dummy tif at {path}")

if __name__ == "__main__":
    dummy_input = "dummy_input.tif"
    dummy_output = "dummy_output.tif"
    model_path = r"C:\Users\Lenovo\Downloads\Best student zoomed out"
    
    create_dummy_tif(dummy_input)
    
    # Run the inference script
    cmd = [
        "python", "inference_tif.py",
        "--input", dummy_input,
        "--model", model_path,
        "--output", dummy_output,
        "--tile_size", "256",
        "--overlap", "32"
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        
        if os.path.exists(dummy_output):
            with rasterio.open(dummy_output) as src:
                print(f"Success! Output mask created with shape: {src.shape}")
        else:
            print("Error: Output file not created.")
            
    except subprocess.CalledProcessError as e:
        print("Command failed with error:")
        print(e.stdout)
        print(e.stderr)
    finally:
        # Cleanup
        if os.path.exists(dummy_input): os.remove(dummy_input)
        # if os.path.exists(dummy_output): os.remove(dummy_output)
