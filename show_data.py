import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

# Paths based on user input
IMG_DIR = Path(r"D:\geo-dataste\zoomed_out_1024\train\images")
MASK_DIR = Path(r"D:\geo-dataste\zoomed_out_1024\train\masks")

def visualize_random_samples(num_samples=3):
    img_files = sorted(list(IMG_DIR.glob("*.npy")))
    if not img_files:
        print(f"No files found in {IMG_DIR}")
        return

    # Select random samples
    indices = np.random.choice(len(img_files), num_samples, replace=False)
    
    fig, axes = plt.subplots(num_samples, 2, figsize=(10, 4 * num_samples))
    if num_samples == 1:
        axes = [axes]

    for i, idx in enumerate(indices):
        img_path = img_files[idx]
        mask_path = MASK_DIR / img_path.name

        if not mask_path.exists():
            print(f"Mask not found for {img_path.name}")
            continue

        img = np.load(img_path)
        mask = np.load(mask_path)

        # Handle shapes (C, H, W) -> (H, W, C)
        if img.ndim == 3:
            img = img.transpose(1, 2, 0)
        
        # Normalize for display if needed (assuming 0-255 uint8 or 0-1 float)
        if img.max() > 1.0:
            img = img.astype(np.uint8)

        # Show image
        axes[i][0].imshow(img)
        axes[i][0].set_title(f"Image: {img_path.name}")
        axes[i][0].axis("off")

        # Show mask with a colormap
        # Classes: 0:BG, 1:Road, 2:Built, 3:Water, 4:Bridge, 5:Railway
        im_mask = axes[i][1].imshow(mask, cmap="tab10", vmin=0, vmax=5)
        axes[i][1].set_title(f"Mask: {mask_path.name}")
        axes[i][1].axis("off")
        plt.colorbar(im_mask, ax=axes[i][1], ticks=range(6))

    plt.tight_layout()
    # Save the visualization to the current directory for the user to "see" if they have an image viewer
    # or just show it if running in an interactive environment.
    output_path = "data_visualization.png"
    plt.savefig(output_path)
    print(f"Visualization saved to {output_path}")
    plt.show()

if __name__ == "__main__":
    visualize_random_samples(3)
