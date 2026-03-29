#!/usr/bin/env python3
"""
=============================================================================
Teacher–Student Training Pipeline — Drone Aerial Segmentation
=============================================================================
Teacher  : Mask2Former (Swin-Base)
Student  : SegFormer (B0) distilled from Teacher

Fixes    : Mask2Former query-based forward, albumentations 2.0.8,
           (C,H,W)→(H,W,C) transpose, Affine, std_range, fill=
Saves    : checkpoint every epoch + best model separately
Status   : ThingSpeak live dashboard after every epoch
Resume   : auto-detects last saved epoch and resumes
=============================================================================
"""

import os
import json
import time
import random
import requests
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast

import albumentations as A
from sklearn.model_selection import train_test_split

from transformers import (
    Mask2FormerForUniversalSegmentation,
    Mask2FormerImageProcessor,
    SegformerForSemanticSegmentation,
    SegformerImageProcessor,
)

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()


# ──────────────────────────────────────────────────────────────────────────────
# THINGSPEAK CONFIG — replace with YOUR values after creating channel
# ──────────────────────────────────────────────────────────────────────────────
THINGSPEAK_WRITE_API  = "RGI3I69ZRJQQ21VO"   # from ThingSpeak → API Keys tab
THINGSPEAK_CHANNEL_ID = "3284819"      # from ThingSpeak channel page
THINGSPEAK_URL        = "https://api.thingspeak.com/update"

# Field mapping (must match exactly what you set in ThingSpeak channel):
# Field 1 → Train Loss
# Field 2 → mIoU
# Field 3 → Road IoU
# Field 4 → Water IoU
# Field 5 → Built-up IoU
# Field 6 → Bridge IoU
# Field 7 → Railway IoU
# Field 8 → Epoch number


def send_thingspeak(stage: str, epoch: int, avg_loss: float, metrics: dict):
    """
    POST one data point to ThingSpeak after each epoch.
    Non-blocking — training continues even if POST fails.
    ThingSpeak free tier: 1 update per 15 seconds minimum.
    """
    payload = {
        "api_key" : THINGSPEAK_WRITE_API,
        "field1"  : round(avg_loss, 6),
        "field2"  : round(metrics.get("mIoU",       0.0), 6),
        "field3"  : round(metrics.get("Road",        0.0), 6),
        "field4"  : round(metrics.get("Water",       0.0), 6),
        "field5"  : round(metrics.get("Built-up",    0.0), 6),
        "field6"  : round(metrics.get("Bridge",      0.0), 6),
        "field7"  : round(metrics.get("Railway",     0.0), 6),
        "field8"  : epoch,
    }
    try:
        r = requests.get(THINGSPEAK_URL, params=payload, timeout=5)
        if r.status_code == 200 and r.text != "0":
            console.print(
                f"  [bold green]📡 ThingSpeak updated — "
                f"stage={stage} epoch={epoch} mIoU={metrics.get('mIoU', 0):.4f}[/bold green]"
            )
        else:
            console.print(f"  [yellow]ThingSpeak warn: {r.status_code} {r.text}[/yellow]")
    except Exception as e:
        console.print(f"  [yellow]ThingSpeak failed (non-fatal): {e}[/yellow]")


# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Config:
    # Paths
    img_dir             : str   = "dataset_6_classes/train/images"
    msk_dir             : str   = "dataset_6_classes/train/masks"
    teacher_out         : str   = "outputs/teacher_mask2former"
    student_out         : str   = "outputs/student_segformer"
    log_path            : str   = "outputs/training_log.json"

    # Pretrained checkpoints
    teacher_ckpt        : str   = "facebook/mask2former-swin-base-ade-semantic"
    student_ckpt        : str   = "nvidia/mit-b0"

    # Data
    num_classes         : int   = 6
    img_size            : int   = 512
    val_split           : float = 0.1
    seed                : int   = 42

    # Teacher
    teacher_epochs      : int   = 15
    teacher_batch       : int   = 2
    teacher_lr          : float = 5e-5
    teacher_wd          : float = 1e-4

    # Student
    student_epochs      : int   = 20
    student_batch       : int   = 4
    student_lr          : float = 1e-4
    student_wd          : float = 1e-4
    kd_temperature      : float = 4.0
    kd_lambda           : float = 1.0

    # clDice weights
    lambda_road         : float = 0.5
    lambda_rail         : float = 0.2
    lambda_bridge       : float = 0.2

    # CE class weights: [bg, road, water, builtup, bridge, railway]
    class_weights       : Tuple = (0.5, 1.5, 1.0, 1.0, 2.0, 2.0)

    # Runtime
    num_workers         : int   = 4
    grad_clip           : float = 1.0
    log_every           : int   = 20
    early_stop_patience : int   = 5


CFG = Config()

CLASS_NAMES = {
    0: "Background",
    1: "Road",
    2: "Water",
    3: "Built-up",
    4: "Bridge",
    5: "Railway",
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ──────────────────────────────────────────────────────────────────────────────
# SEED
# ──────────────────────────────────────────────────────────────────────────────

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ──────────────────────────────────────────────────────────────────────────────
# AUGMENTATIONS — albumentations 2.0.8
# ──────────────────────────────────────────────────────────────────────────────

def get_curriculum_augmentations(img_size: int, epoch: int, total_epochs: int):
    phase = epoch / total_epochs
    if phase < 0.33:
        occlusion_p, max_holes = 0.0, 0
    elif phase < 0.66:
        occlusion_p, max_holes = 0.15, 4
    else:
        occlusion_p, max_holes = 0.30, 8

    transforms = [
        A.RandomResizedCrop(size=(img_size, img_size), scale=(0.5, 1.0), ratio=(0.75, 1.33), p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.5),
        A.Affine(
            translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
            scale=(0.9, 1.1),
            rotate=(-15, 15),
            p=0.4,
        ),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10),
            A.CLAHE(clip_limit=4.0),
        ], p=0.6),
        A.GaussNoise(std_range=(0.01, 0.05), p=0.3),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
    ]

    if occlusion_p > 0 and max_holes > 0:
        transforms.append(
            A.CoarseDropout(
                num_holes_range=(1, max_holes),
                hole_height_range=(16, 32),
                hole_width_range=(16, 32),
                fill=0,
                p=occlusion_p,
            )
        )
    return A.Compose(transforms, is_check_shapes=False)


def get_val_augmentations(img_size: int):
    return A.Compose([
        A.Resize(height=img_size, width=img_size),
    ], is_check_shapes=False)


# ──────────────────────────────────────────────────────────────────────────────
# DATASET
# ──────────────────────────────────────────────────────────────────────────────

class DroneDataset(Dataset):
    def __init__(
        self,
        img_files    : List[str],
        msk_dir      : str,
        mean         : List[float],
        std          : List[float],
        img_size     : int,
        is_train     : bool = True,
        epoch        : int  = None,
        total_epochs : int  = None,
    ):
        self.img_files = img_files
        self.msk_dir   = msk_dir
        self.img_size  = img_size
        self.mean      = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
        self.std       = np.array(std,  dtype=np.float32).reshape(1, 1, 3)
        if is_train and epoch is not None:
            self.augs = get_curriculum_augmentations(img_size, epoch, total_epochs)
        else:
            self.augs = get_val_augmentations(img_size)

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        msk_path = os.path.join(self.msk_dir, os.path.basename(img_path))

        image = np.load(img_path)
        mask  = np.load(msk_path)

        # (C,H,W) → (H,W,C) for albumentations
        if image.ndim == 3 and image.shape[0] in (1, 3, 4):
            image = np.transpose(image, (1, 2, 0))

        # Ensure uint8 [0,255]
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8) if image.max() <= 1.1 else image.astype(np.uint8)

        mask = mask.astype(np.int32)

        out   = self.augs(image=image, mask=mask)
        image = out["image"]
        mask  = out["mask"]

        image = image.astype(np.float32) / 255.0
        image = (image - self.mean) / self.std
        image = np.transpose(image, (2, 0, 1))

        return torch.from_numpy(image), torch.from_numpy(mask.astype(np.int64))


# ──────────────────────────────────────────────────────────────────────────────
# MASK2FORMER FORWARD
# ──────────────────────────────────────────────────────────────────────────────

def mask2former_logits(outputs, target_hw: Tuple):
    """
    Mask2Former returns query-based outputs, not .logits
    class_queries_logits : (B, Q, C+1)  — +1 for no-object
    masks_queries_logits : (B, Q, H', W')
    """
    class_probs = F.softmax(outputs.class_queries_logits, dim=-1)[..., :-1]  # (B,Q,C)
    mask_probs  = outputs.masks_queries_logits.sigmoid()                      # (B,Q,H',W')
    semantic    = torch.einsum("bqc,bqhw->bchw", class_probs, mask_probs)
    return F.interpolate(semantic, size=target_hw, mode="bilinear", align_corners=False)


# ──────────────────────────────────────────────────────────────────────────────
# clDice LOSS
# ──────────────────────────────────────────────────────────────────────────────

def soft_erode(x):
    return torch.min(
        -F.max_pool2d(-x, (3,1), (1,1), (1,0)),
        -F.max_pool2d(-x, (1,3), (1,1), (0,1))
    )

def soft_dilate(x):
    return F.max_pool2d(x, (3,3), (1,1), (1,1))

def soft_open(x):
    return soft_dilate(soft_erode(x))

def soft_skel(x, iters: int = 10):
    img1 = soft_open(x)
    skel = F.relu(x - img1)
    for _ in range(iters):
        x    = soft_erode(x)
        img1 = soft_open(x)
        d    = F.relu(x - img1)
        skel = skel + F.relu(d - skel * d)
    return skel

def cldice(pred, true, smooth: float = 1e-5):
    sp    = soft_skel(pred)
    sl    = soft_skel(true)
    tprec = (sp * true).sum() / (sp.sum() + smooth)
    tsens = (sl * pred).sum() / (sl.sum() + smooth)
    return 1.0 - 2.0 * tprec * tsens / (tprec + tsens + smooth)


def segmentation_loss(logits, labels, cfg: Config):
    device  = logits.device
    weights = torch.tensor(cfg.class_weights, dtype=torch.float32, device=device)
    ce      = F.cross_entropy(logits, labels.long(), weight=weights)
    probs   = torch.softmax(logits, dim=1)

    def _topo(c):
        return cldice(probs[:, c:c+1], (labels == c).unsqueeze(1).float())

    loss_road, loss_rail, loss_bridge = _topo(1), _topo(5), _topo(4)

    total = (ce
             + cfg.lambda_road   * loss_road
             + cfg.lambda_rail   * loss_rail
             + cfg.lambda_bridge * loss_bridge)
    return total, {
        "ce"    : ce.item(),
        "road"  : loss_road.item(),
        "rail"  : loss_rail.item(),
        "bridge": loss_bridge.item(),
    }


def kd_loss(s_logits, t_logits, T: float):
    s_log = F.log_softmax(s_logits / T, dim=1)
    t_p   = F.softmax(t_logits   / T, dim=1)
    return F.kl_div(s_log, t_p, reduction="batchmean") * (T * T)


# ──────────────────────────────────────────────────────────────────────────────
# METRICS
# ──────────────────────────────────────────────────────────────────────────────

def per_class_iou(pred_np, gt_np, n: int = 6):
    out = {}
    for c in range(n):
        inter = ((pred_np == c) & (gt_np == c)).sum()
        union = ((pred_np == c) | (gt_np == c)).sum()
        if union > 0:
            out[c] = inter / union
    return out


@torch.no_grad()
def evaluate(model, val_loader, cfg: Config, is_teacher: bool = False):
    model.eval()
    accum = {c: [] for c in range(cfg.num_classes)}
    for images, masks in val_loader:
        images  = images.to(DEVICE)
        outputs = model(pixel_values=images)
        if is_teacher:
            logits = mask2former_logits(outputs, masks.shape[-2:])
        else:
            logits = F.interpolate(
                outputs.logits, size=masks.shape[-2:],
                mode="bilinear", align_corners=False
            )
        preds = logits.argmax(dim=1).cpu().numpy()
        gts   = masks.numpy()
        for i in range(len(preds)):
            for c, v in per_class_iou(preds[i], gts[i], cfg.num_classes).items():
                accum[c].append(v)

    result = {
        CLASS_NAMES[c]: float(np.mean(accum[c]))
        for c in range(cfg.num_classes) if accum[c]
    }
    result["mIoU"] = float(np.mean(list(result.values()))) if result else 0.0
    return result


def print_metrics(metrics: dict, title: str):
    t = Table(title=title, show_header=True)
    t.add_column("Class", style="white")
    t.add_column("IoU",   style="cyan")
    for k, v in metrics.items():
        style = "bold yellow" if k == "mIoU" else ""
        t.add_row(k, f"{v:.4f}", style=style)
    console.print(t)


# ──────────────────────────────────────────────────────────────────────────────
# CHECKPOINT HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def save_epoch_checkpoint(model, processor, base_dir: str, epoch: int):
    """Save a full checkpoint for every epoch — enables power-cut recovery."""
    epoch_dir = os.path.join(base_dir, f"epoch-{epoch:03d}")
    os.makedirs(epoch_dir, exist_ok=True)
    model.save_pretrained(epoch_dir)
    processor.save_pretrained(epoch_dir)
    return epoch_dir


def find_last_epoch_checkpoint(base_dir: str):
    """
    Scan base_dir for epoch-NNN folders and return the highest N found.
    Returns (epoch_number, path) or (0, None) if none found.
    """
    if not os.path.exists(base_dir):
        return 0, None
    candidates = []
    for d in os.listdir(base_dir):
        if d.startswith("epoch-") and os.path.isdir(os.path.join(base_dir, d)):
            try:
                candidates.append(int(d.split("-")[1]))
            except ValueError:
                pass
    if not candidates:
        return 0, None
    last = max(candidates)
    return last, os.path.join(base_dir, f"epoch-{last:03d}")


# ──────────────────────────────────────────────────────────────────────────────
# TRAIN TEACHER
# ──────────────────────────────────────────────────────────────────────────────

def train_teacher(cfg: Config):
    console.print(Panel.fit(
        "[bold blue]Stage 1 — TEACHER (Mask2Former Swin-Base)[/bold blue]\n"
        "[cyan]Forward : mask2former_logits — einsum(class_probs × mask_probs)[/cyan]\n"
        "[cyan]Loss    : Weighted CE + clDice(road/rail/bridge)[/cyan]\n"
        "[cyan]Saves   : checkpoint every epoch + best model separately[/cyan]\n"
        "[cyan]Status  : ThingSpeak after every epoch[/cyan]"
    ))

    # ── Auto-resume from last epoch if available ──────────────────────────────
    last_epoch, last_ckpt = find_last_epoch_checkpoint(cfg.teacher_out)
    if last_ckpt:
        console.print(f"[yellow]Resuming Teacher from epoch {last_epoch} — {last_ckpt}[/yellow]")
        processor = Mask2FormerImageProcessor.from_pretrained(last_ckpt, do_reduce_labels=False)
        model     = Mask2FormerForUniversalSegmentation.from_pretrained(
            last_ckpt, num_labels=cfg.num_classes, ignore_mismatched_sizes=True
        ).to(DEVICE)
    else:
        console.print(f"[green]Starting Teacher from scratch — {cfg.teacher_ckpt}[/green]")
        processor = Mask2FormerImageProcessor.from_pretrained(
            cfg.teacher_ckpt, do_reduce_labels=False
        )
        model = Mask2FormerForUniversalSegmentation.from_pretrained(
            cfg.teacher_ckpt, num_labels=cfg.num_classes, ignore_mismatched_sizes=True
        ).to(DEVICE)

    console.print(
        f"[green]Trainable params: "
        f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}[/green]"
    )

    mean = processor.image_mean
    std  = processor.image_std

    all_imgs = sorted([
        os.path.join(cfg.img_dir, f)
        for f in os.listdir(cfg.img_dir) if f.endswith(".npy")
    ])
    train_imgs, val_imgs = train_test_split(
        all_imgs, test_size=cfg.val_split, random_state=cfg.seed
    )
    console.print(
        f"[green]Total: {len(all_imgs)}  Train: {len(train_imgs)}  Val: {len(val_imgs)}[/green]"
    )

    val_ds     = DroneDataset(val_imgs, cfg.msk_dir, mean, std, cfg.img_size, is_train=False)
    val_loader = DataLoader(
        val_ds, batch_size=cfg.teacher_batch, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.teacher_lr, weight_decay=cfg.teacher_wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.teacher_epochs, eta_min=1e-6
    )
    # Advance scheduler to match resumed epoch
    for _ in range(last_epoch):
        scheduler.step()

    scaler     = GradScaler()
    best_miou  = 0.0
    no_improve = 0

    # Load existing log if resuming
    log = {"teacher": [], "student": []}
    if os.path.exists(cfg.log_path):
        with open(cfg.log_path) as f:
            log = json.load(f)

    os.makedirs(cfg.teacher_out, exist_ok=True)
    start_epoch = last_epoch + 1

    for epoch in range(start_epoch, cfg.teacher_epochs + 1):
        console.print(
            f"\n[bold]━━━ Teacher Epoch {epoch}/{cfg.teacher_epochs}  "
            f"LR={scheduler.get_last_lr()[0]:.2e} ━━━[/bold]"
        )
        t0 = time.time()

        train_ds = DroneDataset(
            train_imgs, cfg.msk_dir, mean, std, cfg.img_size,
            is_train=True, epoch=epoch, total_epochs=cfg.teacher_epochs
        )
        train_loader = DataLoader(
            train_ds, batch_size=cfg.teacher_batch, shuffle=True,
            num_workers=cfg.num_workers, pin_memory=True, drop_last=True
        )

        model.train()
        epoch_loss = 0.0

        for step, (images, masks) in enumerate(train_loader, start=1):
            images = images.to(DEVICE)
            masks  = masks.to(DEVICE)

            optimizer.zero_grad()
            with autocast():
                outputs = model(pixel_values=images)
                logits  = mask2former_logits(outputs, masks.shape[-2:])
                loss, bd = segmentation_loss(logits, masks, cfg)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

            if step % cfg.log_every == 0:
                console.print(
                    f"  step {step:04d} | loss={loss.item():.4f}  "
                    f"ce={bd['ce']:.4f}  road={bd['road']:.4f}  "
                    f"rail={bd['rail']:.4f}  brdg={bd['bridge']:.4f}"
                )

        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        elapsed  = time.time() - t0
        console.print(f"  Avg loss: {avg_loss:.4f}   Time: {elapsed:.1f}s")

        metrics = evaluate(model, val_loader, cfg, is_teacher=True)
        print_metrics(metrics, f"Teacher Epoch {epoch} — Validation IoU")

        # ── Save every epoch ──────────────────────────────────────────────────
        epoch_dir = save_epoch_checkpoint(model, processor, cfg.teacher_out, epoch)
        console.print(f"  [cyan]Epoch checkpoint saved → {epoch_dir}[/cyan]")

        # ── Save best separately ──────────────────────────────────────────────
        if metrics["mIoU"] > best_miou:
            best_miou  = metrics["mIoU"]
            no_improve = 0
            best_dir   = os.path.join(cfg.teacher_out, "best")
            os.makedirs(best_dir, exist_ok=True)
            model.save_pretrained(best_dir)
            processor.save_pretrained(best_dir)
            console.print(
                f"  [bold green]💾 Best Teacher mIoU={best_miou:.4f} → {best_dir}[/bold green]"
            )
        else:
            no_improve += 1
            console.print(
                f"  [yellow]No improvement {no_improve}/{cfg.early_stop_patience}[/yellow]"
            )

        # ── ThingSpeak update ─────────────────────────────────────────────────
        send_thingspeak("teacher", epoch, avg_loss, metrics)

        # ── Log to JSON ───────────────────────────────────────────────────────
        log["teacher"].append({
            "epoch": epoch, "loss": avg_loss, "time_s": elapsed, **metrics
        })
        Path(cfg.log_path).parent.mkdir(parents=True, exist_ok=True)
        with open(cfg.log_path, "w") as f:
            json.dump(log, f, indent=2)

        if no_improve >= cfg.early_stop_patience:
            console.print(f"[red]Early stopping at epoch {epoch}[/red]")
            break

    console.print(f"[bold green]Teacher done. Best mIoU={best_miou:.4f}[/bold green]")

    # Load best weights for distillation
    best_dir = os.path.join(cfg.teacher_out, "best")
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        best_dir, num_labels=cfg.num_classes, ignore_mismatched_sizes=True
    ).to(DEVICE)
    return processor, model, val_imgs, log


# ──────────────────────────────────────────────────────────────────────────────
# TRAIN STUDENT
# ──────────────────────────────────────────────────────────────────────────────

def train_student(cfg: Config, teacher_model, val_imgs_teacher, log: dict):
    console.print(Panel.fit(
        "[bold magenta]Stage 2 — STUDENT (SegFormer B0)[/bold magenta]\n"
        "[cyan]Loss  : Supervised CE+clDice  +  KD from frozen Teacher[/cyan]\n"
        "[cyan]KD    : KL divergence T=4.0[/cyan]\n"
        "[cyan]Saves : checkpoint every epoch + best model separately[/cyan]\n"
        "[cyan]Status: ThingSpeak after every epoch[/cyan]"
    ))

    # ── Auto-resume ───────────────────────────────────────────────────────────
    last_epoch, last_ckpt = find_last_epoch_checkpoint(cfg.student_out)
    if last_ckpt:
        console.print(f"[yellow]Resuming Student from epoch {last_epoch} — {last_ckpt}[/yellow]")
        s_processor = SegformerImageProcessor.from_pretrained(last_ckpt, do_reduce_labels=False)
        student     = SegformerForSemanticSegmentation.from_pretrained(
            last_ckpt, num_labels=cfg.num_classes, ignore_mismatched_sizes=True
        ).to(DEVICE)
    else:
        console.print(f"[green]Starting Student from scratch — {cfg.student_ckpt}[/green]")
        s_processor = SegformerImageProcessor.from_pretrained(
            cfg.student_ckpt, do_reduce_labels=False
        )
        student = SegformerForSemanticSegmentation.from_pretrained(
            cfg.student_ckpt, num_labels=cfg.num_classes, ignore_mismatched_sizes=True
        ).to(DEVICE)

    console.print(
        f"[green]Student trainable params: "
        f"{sum(p.numel() for p in student.parameters() if p.requires_grad):,}[/green]"
    )

    t_processor = Mask2FormerImageProcessor.from_pretrained(
        os.path.join(cfg.teacher_out, "best"), do_reduce_labels=False
    )

    teacher_model.eval()
    for p in teacher_model.parameters():
        p.requires_grad = False

    all_imgs = sorted([
        os.path.join(cfg.img_dir, f)
        for f in os.listdir(cfg.img_dir) if f.endswith(".npy")
    ])
    train_imgs, val_imgs = train_test_split(
        all_imgs, test_size=cfg.val_split, random_state=cfg.seed
    )

    s_mean, s_std = s_processor.image_mean, s_processor.image_std
    t_mean, t_std = t_processor.image_mean, t_processor.image_std

    val_ds     = DroneDataset(val_imgs, cfg.msk_dir, s_mean, s_std, cfg.img_size, is_train=False)
    val_loader = DataLoader(
        val_ds, batch_size=cfg.student_batch, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True
    )

    optimizer = torch.optim.AdamW(student.parameters(), lr=cfg.student_lr, weight_decay=cfg.student_wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.student_epochs, eta_min=1e-6
    )
    for _ in range(last_epoch):
        scheduler.step()

    scaler     = GradScaler()
    best_miou  = 0.0
    no_improve = 0
    start_epoch = last_epoch + 1

    os.makedirs(cfg.student_out, exist_ok=True)

    for epoch in range(start_epoch, cfg.student_epochs + 1):
        console.print(
            f"\n[bold]━━━ Student Epoch {epoch}/{cfg.student_epochs}  "
            f"LR={scheduler.get_last_lr()[0]:.2e} ━━━[/bold]"
        )
        t0 = time.time()

        s_train_ds = DroneDataset(
            train_imgs, cfg.msk_dir, s_mean, s_std, cfg.img_size,
            is_train=True, epoch=epoch, total_epochs=cfg.student_epochs
        )
        t_train_ds = DroneDataset(
            train_imgs, cfg.msk_dir, t_mean, t_std, cfg.img_size, is_train=False
        )

        s_loader = DataLoader(
            s_train_ds, batch_size=cfg.student_batch, shuffle=False,
            num_workers=cfg.num_workers, pin_memory=True, drop_last=True
        )
        t_loader = DataLoader(
            t_train_ds, batch_size=cfg.student_batch, shuffle=False,
            num_workers=cfg.num_workers, pin_memory=True, drop_last=True
        )

        student.train()
        epoch_loss = 0.0

        for step, ((s_imgs, masks), (t_imgs, _)) in enumerate(
            zip(s_loader, t_loader), start=1
        ):
            s_imgs = s_imgs.to(DEVICE)
            t_imgs = t_imgs.to(DEVICE)
            masks  = masks.to(DEVICE)

            optimizer.zero_grad()
            with autocast():
                with torch.no_grad():
                    t_out    = teacher_model(pixel_values=t_imgs)
                    t_logits = mask2former_logits(t_out, masks.shape[-2:])

                s_out    = student(pixel_values=s_imgs)
                s_logits = F.interpolate(
                    s_out.logits, size=masks.shape[-2:],
                    mode="bilinear", align_corners=False
                )

                sup_loss, bd = segmentation_loss(s_logits, masks, cfg)
                kd           = kd_loss(s_logits, t_logits, cfg.kd_temperature)
                total_loss   = sup_loss + cfg.kd_lambda * kd

            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(student.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += total_loss.item()

            if step % cfg.log_every == 0:
                console.print(
                    f"  step {step:04d} | total={total_loss.item():.4f}  "
                    f"sup={sup_loss.item():.4f}  kd={kd.item():.4f}  "
                    f"ce={bd['ce']:.4f}  road={bd['road']:.4f}"
                )

        scheduler.step()
        avg_loss = epoch_loss / len(s_loader)
        elapsed  = time.time() - t0
        console.print(f"  Avg loss: {avg_loss:.4f}   Time: {elapsed:.1f}s")

        metrics = evaluate(student, val_loader, cfg, is_teacher=False)
        print_metrics(metrics, f"Student Epoch {epoch} — Validation IoU")

        # ── Save every epoch ──────────────────────────────────────────────────
        epoch_dir = save_epoch_checkpoint(student, s_processor, cfg.student_out, epoch)
        console.print(f"  [cyan]Epoch checkpoint saved → {epoch_dir}[/cyan]")

        # ── Save best separately ──────────────────────────────────────────────
        if metrics["mIoU"] > best_miou:
            best_miou  = metrics["mIoU"]
            no_improve = 0
            best_dir   = os.path.join(cfg.student_out, "best")
            os.makedirs(best_dir, exist_ok=True)
            student.save_pretrained(best_dir)
            s_processor.save_pretrained(best_dir)
            console.print(
                f"  [bold green]💾 Best Student mIoU={best_miou:.4f} → {best_dir}[/bold green]"
            )
        else:
            no_improve += 1
            console.print(
                f"  [yellow]No improvement {no_improve}/{cfg.early_stop_patience}[/yellow]"
            )

        # ── ThingSpeak update ─────────────────────────────────────────────────
        send_thingspeak("student", epoch, avg_loss, metrics)

        # ── Log to JSON ───────────────────────────────────────────────────────
        log["student"].append({
            "epoch": epoch, "loss": avg_loss, "time_s": elapsed, **metrics
        })
        with open(cfg.log_path, "w") as f:
            json.dump(log, f, indent=2)

        if no_improve >= cfg.early_stop_patience:
            console.print(f"[red]Early stopping at epoch {epoch}[/red]")
            break

    console.print(f"[bold green]Student done. Best mIoU={best_miou:.4f}[/bold green]")
    return s_processor, student


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    seed_everything(CFG.seed)
    Path("outputs").mkdir(parents=True, exist_ok=True)
    with open("outputs/train_config.json", "w") as f:
        json.dump(asdict(CFG), f, indent=2)

    console.print(Panel.fit(
        "[bold]🛰  Drone Segmentation — Teacher + Student[/bold]\n"
        f"Device    : [green]{DEVICE}[/green]\n"
        f"Teacher   : [cyan]{CFG.teacher_ckpt}[/cyan]\n"
        f"Student   : [magenta]{CFG.student_ckpt}[/magenta]\n"
        f"Classes   : {list(CLASS_NAMES.values())}\n"
        f"Dashboard : [link=https://thingspeak.com/channels/{THINGSPEAK_CHANNEL_ID}]"
        f"ThingSpeak Channel {THINGSPEAK_CHANNEL_ID}[/link]"
    ))

    t_proc, t_model, val_imgs, log = train_teacher(CFG)
    s_proc, s_model = train_student(CFG, t_model, val_imgs, log)

    console.print(Panel.fit(
        "[bold green]✅ Pipeline complete[/bold green]\n"
        f"Teacher best → [cyan]{CFG.teacher_out}/best[/cyan]\n"
        f"Student best → [magenta]{CFG.student_out}/best[/magenta]\n"
        f"Log          → [white]{CFG.log_path}[/white]"
    ))

    import subprocess
    subprocess.run(["python", "visualize_results.py"], check=True)


if __name__ == "__main__":
    main()
