"""
train_mask2former.py
====================
Fine-tune Mask2Former on the geo-spatial segmentation dataset.

HuggingFace Hub ID: "facebook/mask2former-swin-tiny-cityscapes-semantic"

Mask2Former formulates segmentation as a mask classification problem.
It handles semantic scaling internally and returns the bipartite matching loss.
Because the labels convert to lists of boolean masks based on presence, a custom `collate_fn` is required.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from tqdm import tqdm

from transformers import (
    Mask2FormerImageProcessor,
    Mask2FormerForUniversalSegmentation,
)


# ── Dataset ───────────────────────────────────────────────────────────────────

class GeoDatasetMask2Former(Dataset):
    """
    Loads pre-tiled .npy patches.
    Images : (C, H, W) float32 normalisation happens internally or scaled to [0, 1]
    Masks  : (H, W)    int64   class labels (e.g. 0 to 5)
    """

    def __init__(self, root_dir, processor: Mask2FormerImageProcessor):
        self.img_dir   = Path(root_dir) / "images"
        self.mask_dir  = Path(root_dir) / "masks"
        self.img_files = sorted(self.img_dir.glob("*.npy"))
        self.processor = processor

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path  = self.img_files[idx]
        mask_path = self.mask_dir / img_path.name

        image = np.load(img_path).astype(np.float32) / 255.0   # (C, H, W)  [0,1]
        mask  = np.load(mask_path).astype(np.int64)            # (H, W)

        # Ensure exactly 3 channels
        image = image[:3]

        # Processor requires segmentation_maps for training labels to generate the binary masks
        enc = self.processor(
            images=image,
            segmentation_maps=mask,
            return_tensors="pt",
            do_rescale=False,   # already [0,1]
        )

        # processor returns lists with 1 item because we passed 1 image
        return {
            "pixel_values": enc["pixel_values"].squeeze(0),
            "class_labels": enc["class_labels"][0],
            "mask_labels": enc["mask_labels"][0],
            "original_mask": torch.from_numpy(mask)
        }


# ── Collate Function ──────────────────────────────────────────────────────────

def collate_fn(batch):
    """
    Mask2Former outputs lists for class_labels and mask_labels because images 
    can have a variable number of classes present in them.
    We stack pixel_values and original_masks, but keep labels as lists.
    """
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    class_labels = [item["class_labels"] for item in batch]
    mask_labels  = [item["mask_labels"] for item in batch]
    original_masks = torch.stack([item["original_mask"] for item in batch])
    
    return {
        "pixel_values": pixel_values,
        "class_labels": class_labels,
        "mask_labels": mask_labels,
        "original_masks": original_masks
    }


# ── Mean IoU ──────────────────────────────────────────────────────────────────

def mean_iou(preds: torch.Tensor, targets: torch.Tensor, num_classes: int) -> float:
    preds   = preds.view(-1)
    targets = targets.view(-1)
    ious = []
    for cls in range(num_classes):
        tp    = ((preds == cls) & (targets == cls)).sum().item()
        fp    = ((preds == cls) & (targets != cls)).sum().item()
        fn    = ((preds != cls) & (targets == cls)).sum().item()
        denom = tp + fp + fn
        if denom > 0:
            ious.append(tp / denom)
    return float(np.mean(ious)) if ious else 0.0


# ── Training ──────────────────────────────────────────────────────────────────

def train():
    # ── Config ──────────────────────────────────────────────────────────────
    DATA_DIR    = r"E:\dataset_stride_384\train"
    BATCH_SIZE  = 2           
    LR          = 5e-5        
    EPOCHS      = 30
    NUM_CLASSES = 6           # 0=BG, 1=Road, 2=Built-up, 3=Water, 4=Bridge, 5=Railway
    SAVE_PATH   = "best_mask2former.pth"
    DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    USE_AMP     = torch.cuda.is_available()

    # Base model name
    MODEL_ID = "facebook/mask2former-swin-tiny-cityscapes-semantic"
    
    # ── Processor ───────────────────────────────────────────────────────────
    # We set ignore_index=255 to properly mask out boundaries if applicable,
    # though our GeoDataset doesn't natively use 255.
    processor = Mask2FormerImageProcessor.from_pretrained(
        MODEL_ID,
        do_resize=False,        
        do_rescale=False,       
        do_normalize=True,      
        ignore_index=255        
    )

    # ── Data ────────────────────────────────────────────────────────────────
    # Fallback if specific stride dir doesn't exist
    if not os.path.exists(DATA_DIR):
        DATA_DIR = r"E:\dataset\train"

    dataset = GeoDatasetMask2Former(DATA_DIR, processor)
    
    if len(dataset) == 0:
        print(f"No data found in {DATA_DIR}. Make sure you generate the dataset first.")
        return

    train_size = int(0.8 * len(dataset))
    val_size   = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, drop_last=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, collate_fn=collate_fn
    )

    H, W = dataset[0]["pixel_values"].shape[1], dataset[0]["pixel_values"].shape[2]
    print(f"Tiles        : {len(dataset)}  (train {train_size} | val {val_size})")
    print(f"Tile size    : {H} × {W}")
    print(f"Device       : {DEVICE}  |  AMP: {USE_AMP}")

    # ── Model ───────────────────────────────────────────────────────────────
    id2label = {i: str(i) for i in range(NUM_CLASSES)}
    label2id = {str(i): i for i in range(NUM_CLASSES)}

    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        MODEL_ID,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,   # Required to replace decode head for NUM_CLASSES
    )
    model.to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Params       : {n_params:.1f}M\n")

    # ── Optimizer ───────────────────────────────────────────────────────────
    # Basic AdamW is fine for Mask2Former, though backbone LR could be smaller if desired
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

    total_steps  = EPOCHS * len(train_loader)
    warmup_steps = max(1, int(0.05 * total_steps))

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    scaler    = torch.amp.GradScaler("cuda", enabled=USE_AMP)

    best_val_loss = float("inf")
    
    # ── Loop ────────────────────────────────────────────────────────────────
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:02d}/{EPOCHS} [Train]")

        for batch in pbar:
            pixel_values = batch["pixel_values"].to(DEVICE)
            class_labels = [labels.to(DEVICE) for labels in batch["class_labels"]]
            mask_labels  = [labels.to(DEVICE) for labels in batch["mask_labels"]]

            optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=USE_AMP):
                outputs = model(
                    pixel_values=pixel_values,
                    class_labels=class_labels,
                    mask_labels=mask_labels
                )
                loss = outputs.loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            train_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

        avg_train_loss = train_loss / len(train_loader)

        # ── Validate ────────────────────────────────────────────────────────
        model.eval()
        val_loss    = 0.0
        all_preds, all_targets = [], []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1:02d}/{EPOCHS} [Val]  "):
                pixel_values   = batch["pixel_values"].to(DEVICE)
                class_labels   = [labels.to(DEVICE) for labels in batch["class_labels"]]
                mask_labels    = [labels.to(DEVICE) for labels in batch["mask_labels"]]
                original_masks = batch["original_masks"]

                with torch.amp.autocast("cuda", enabled=USE_AMP):
                    outputs = model(
                        pixel_values=pixel_values,
                        class_labels=class_labels,
                        mask_labels=mask_labels
                    )
                    loss = outputs.loss
                
                val_loss += loss.item()

                # Get predicted masks. 
                # Mask2Former outputs predictions in the shape of list of boolean masks.
                # The post_process_semantic_segmentation method collapses these queries 
                # into standard semantic map classes.
                target_sizes = [mask.shape for mask in original_masks]
                pred_maps = processor.post_process_semantic_segmentation(outputs, target_sizes=target_sizes)
                
                for pred, target in zip(pred_maps, original_masks):
                    all_preds.append(pred.cpu())
                    all_targets.append(target.cpu())

        avg_val_loss = val_loss / len(val_loader)
        miou = mean_iou(torch.cat(all_preds), torch.cat(all_targets), NUM_CLASSES)
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch+1:02d}/{EPOCHS}  "
            f"Train: {avg_train_loss:.4f}  "
            f"Val: {avg_val_loss:.4f}  "
            f"mIoU: {miou:.4f}  "
            f"LR: {current_lr:.2e}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"  ✓ Saved best model → {SAVE_PATH}  (val loss {best_val_loss:.4f})")

if __name__ == "__main__":
    train()
