#!/usr/bin/env python3
"""
train_mixed.py — Final hackathon-optimized version

Fixes:
  - CosineAnnealingLR resume bug (initial_lr injection)
  - Deprecated albumentations args
  - OneCycleLR replaced with resumable CosineAnnealingLR + manual warmup

Optimizations:
  - AMP (mixed precision) for 2x speed
  - KL-divergence KD (better than MSE)
  - Label smoothing on CE
  - Differential LR (backbone vs head)
  - EMA model for validation (typically +1-3% mIoU)
  - Manual warmup for first 2 epochs
  - persistent_workers + num_workers=4
"""

import os, gc, json, random, time
from copy import deepcopy
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
from transformers import (
    Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor,
    SegformerForSemanticSegmentation, SegformerImageProcessor,
)
import albumentations as A

# ─────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────

ZOOMED_IMG_DIR = "zoomed_out_1024/train/images"
ZOOMED_MSK_DIR = "zoomed_out_1024/train/masks"
ORIG_IMG_DIR   = "dataset_6_classes/train/images"
ORIG_MSK_DIR   = "dataset_6_classes/train/masks"

TEACHER_DIR      = "outputs/zoomed_teacher/best"
STUDENT_DIR      = "outputs/zoomed_student/best"
OUT_DIR          = "outputs/mixed_student"
RESUME_FILE      = "outputs/mixed_resume.json"
LLM_HISTORY_FILE = "outputs/mixed_llm_history.json"

ZOOMED_IMG_DIR = "zoomed_out_1024/train/images"
ZOOMED_MSK_DIR = "zoomed_out_1024/train/masks"
ORIG_IMG_DIR   = "dataset_6_classes/train/images"
ORIG_MSK_DIR   = "dataset_6_classes/train/masks"

IMG_DIR = os.path.join(ZOOMED_IMG_DIR, ORIG_IMG_DIR)
MSK_DIR = os.path.join(ZOOMED_MSK_DIR, ORIG_MSK_DIR)
NUM_CLASSES = 4
CLASS_NAMES = ["Water", "Road", "Built-up", "Background"]

EPOCHS        = 40
TOTAL_SAMPLES = 3000
BATCH_SIZE    = 16
LR            = 1e-4
WARMUP_EPOCHS = 2

KD_TEMP    = 4.0
ALPHA      = 0.7
CE_WEIGHTS = [2.5, 3.0, 2.5, 0.5]
DICE_WEIGHT = 0.1
CE_WEIGHTS = [2.5, 3.0, 2.5, 0.5]
EMA_DECAY  = 0.999

VAL_SPLIT = 0.1
SEED      = 42
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP   = torch.cuda.is_available()

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs("outputs", exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ─────────────────────────────────────────────────
# EMA
# ─────────────────────────────────────────────────

class EMA:
    """Exponential Moving Average of model weights — stored on CPU."""
    def __init__(self, model, decay=0.999):
        self.decay  = decay
        self.shadow = {k: v.cpu().clone().float()
                       for k, v in model.state_dict().items()}
        self._backup = None

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            self.shadow[k] = (self.decay * self.shadow[k]
                              + (1.0 - self.decay) * v.cpu().float())

    @torch.no_grad()
    def apply(self, model):
        """Load EMA weights into model for evaluation."""
        self._backup = {k: v.clone() for k, v in model.state_dict().items()}
        ema_state = {k: v.to(model.device if hasattr(model, 'device')
                              else next(model.parameters()).device)
                     for k, v in self.shadow.items()}
        model.load_state_dict(ema_state, strict=False)

    @torch.no_grad()
    def restore(self, model):
        """Restore original weights after EMA evaluation."""
        if self._backup:
            model.load_state_dict(self._backup)
            self._backup = None

# ─────────────────────────────────────────────────
# AUGMENTATION — fixed deprecated args
# ─────────────────────────────────────────────────

TRAIN_AUG = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.RandomRotate90(p=0.5),
    A.Affine(scale=(0.8, 1.2),
             translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
             rotate=(-30, 30), p=0.5),
    A.OneOf([
        A.GridDistortion(num_steps=5, distort_limit=0.3, p=1.0),
        A.ElasticTransform(alpha=120, sigma=6.0, p=1.0),
        A.OpticalDistortion(distort_limit=0.2, p=1.0),
    ], p=0.4),
    A.OneOf([
        A.RandomBrightnessContrast(0.3, 0.3, p=1.0),
        A.HueSaturationValue(20, 40, 20, p=1.0),
        A.RandomGamma(gamma_limit=(70, 130), p=1.0),
    ], p=0.5),
    A.GaussNoise(p=0.3),
    A.CoarseDropout(p=0.3),
    A.RandomShadow(p=0.2),
    A.CLAHE(clip_limit=4.0, p=0.3),
])

VAL_AUG = A.Compose([])

# ─────────────────────────────────────────────────
# MASK REMAP
# ─────────────────────────────────────────────────

def remap_mask(mask_np):
    out = mask_np.astype(np.int64).copy()
    out[out > 3] = 3
    return out

# ─────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────

class MixedDroneDataset(Dataset):
    def __init__(self, pairs, aug=None):
        self.pairs = pairs
        self.aug   = aug

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, msk_path = self.pairs[idx]
        img = np.load(img_path)
        msk = np.load(msk_path)

        if img.ndim == 3 and img.shape[0] in (1, 3, 4):
            img = np.transpose(img, (1, 2, 0))
        img = img.astype(np.uint8)
        msk = remap_mask(msk).astype(np.int64)

        if self.aug:
            out = self.aug(image=img, mask=msk.astype(np.uint8))
            img = out["image"]
            msk = out["mask"].astype(np.int64)

        return img, msk


def collate_fn(batch):
    imgs, msks = zip(*batch)
    return list(imgs), list(msks)


def _gather_pairs(img_dir, msk_dir):
    files = sorted([f for f in os.listdir(img_dir) if f.endswith(".npy")])
    return [(os.path.join(img_dir, f), os.path.join(msk_dir, f))
            for f in files if os.path.exists(os.path.join(msk_dir, f))]


def build_loaders_for_epoch():
    zoomed_pairs = _gather_pairs(ZOOMED_IMG_DIR, ZOOMED_MSK_DIR)
    orig_pairs   = _gather_pairs(ORIG_IMG_DIR,   ORIG_MSK_DIR)

    zoomed_ratio = random.uniform(0.4, 0.7)
    n_zoomed = int(TOTAL_SAMPLES * zoomed_ratio)
    n_orig   = TOTAL_SAMPLES - n_zoomed
    n_zoomed = min(n_zoomed, int(len(zoomed_pairs) * (1 - VAL_SPLIT)))
    n_orig   = min(n_orig,   int(len(orig_pairs)   * (1 - VAL_SPLIT)))

    random.shuffle(zoomed_pairs)
    random.shuffle(orig_pairs)

    n_val_zoomed = max(1, int(len(zoomed_pairs) * VAL_SPLIT))
    val_pairs    = zoomed_pairs[:n_val_zoomed]

    train_zoomed = zoomed_pairs[n_val_zoomed: n_val_zoomed + n_zoomed]
    train_orig   = orig_pairs[:n_orig]
    train_pairs  = train_zoomed + train_orig
    random.shuffle(train_pairs)

    print(f"  Epoch mix → zoomed: {len(train_zoomed)}  "
          f"original: {len(train_orig)}  "
          f"total: {len(train_pairs)}  val: {len(val_pairs)}")

    train_loader = DataLoader(
        MixedDroneDataset(train_pairs, aug=TRAIN_AUG),
        batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4, pin_memory=True,
        persistent_workers=True, collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        MixedDroneDataset(val_pairs, aug=VAL_AUG),
        batch_size=4, shuffle=False,
        num_workers=2, pin_memory=True,
        collate_fn=collate_fn,
    )
    return train_loader, val_loader

# ─────────────────────────────────────────────────
# PREPROCESS HELPERS
# ─────────────────────────────────────────────────

def preprocess(proc, imgs, device):
    pil_list = [Image.fromarray(img) for img in imgs]
    return proc(images=pil_list, return_tensors="pt")["pixel_values"].to(device)


def labels_tensor(msks, device):
    return torch.from_numpy(np.stack([
        m if isinstance(m, np.ndarray) else np.array(m) for m in msks
    ])).long().to(device)

# ─────────────────────────────────────────────────
# TEACHER FORWARD (4-class)
# ─────────────────────────────────────────────────

@torch.no_grad()
def teacher_logits_4class(model_t, pixel_values):
    outputs     = model_t(pixel_values=pixel_values)
    class_probs = F.softmax(outputs.class_queries_logits, dim=-1)[..., :-1]
    mask_probs  = outputs.masks_queries_logits.sigmoid()
    sem6 = torch.einsum("bqc,bqhw->bchw", class_probs, mask_probs)
    sem6 = F.interpolate(sem6, size=(512, 512),
                         mode="bilinear", align_corners=False)
    bg = sem6[:, 3:4] + sem6[:, 4:5] + sem6[:, 5:6]
    return torch.cat([sem6[:, 0:1], sem6[:, 1:2], sem6[:, 2:3], bg], dim=1)

# ─────────────────────────────────────────────────
# VALIDATION
# ─────────────────────────────────────────────────

@torch.no_grad()
def validate(model, proc, val_loader):
    model.eval()
    iou_sum = np.zeros(NUM_CLASSES)
    iou_cnt = np.zeros(NUM_CLASSES)

    for imgs, msks in val_loader:
        gt  = np.stack([m if isinstance(m, np.ndarray)
                        else np.array(m) for m in msks])
        pix = preprocess(proc, imgs, DEVICE)
        with autocast(enabled=USE_AMP):
            logits = model(pixel_values=pix).logits
        logits = F.interpolate(logits.float(), size=gt.shape[-2:],
                               mode="bilinear", align_corners=False)
        pred = logits.argmax(dim=1).cpu().numpy()

        for c in range(NUM_CLASSES):
            inter = np.logical_and(pred == c, gt == c).sum()
            uni   = np.logical_or(pred == c, gt == c).sum()
            if uni > 0:
                iou_sum[c] += inter / uni
                iou_cnt[c] += 1

    iou  = np.divide(iou_sum, np.maximum(iou_cnt, 1))
    miou = float(iou[iou_cnt > 0].mean())
    model.train()
    return miou, iou

# ─────────────────────────────────────────────────
# RESUME
# ─────────────────────────────────────────────────

def load_resume():
    if os.path.exists(RESUME_FILE):
        with open(RESUME_FILE) as f:
            s = json.load(f)
        print(f"  Resumed from epoch {s['epoch']}, "
              f"best mIoU={s['best_miou']:.4f}")
        return s
    return {"epoch": 0, "best_miou": 0.0}


def save_resume(state):
    with open(RESUME_FILE, "w") as f:
        json.dump(state, f, indent=2)

# ─────────────────────────────────────────────────
# LLM AUTOTUNE
# ─────────────────────────────────────────────────

def _load_llm_history():
    if os.path.exists(LLM_HISTORY_FILE):
        with open(LLM_HISTORY_FILE) as f:
            return json.load(f)
    return []


def _save_llm_history(h):
    with open(LLM_HISTORY_FILE, "w") as f:
        json.dump(h, f, indent=2)


def run_autotune(epoch: int, miou: float):
    import llm_helper
    history  = _load_llm_history()
    hist_str = json.dumps(history[-5:], indent=2) if history else ""

    with open(__file__, "r") as f:
        full_script = f.read()

    print("\n[LLM] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"[LLM] Epoch {epoch} complete | mIoU={miou:.4f}")
    print("[LLM] Requesting aggressive suggestion...")

    proposal = llm_helper.suggest_change(full_script, hist_str)
    if proposal is None:
        print("[LLM] No valid proposal.")
        return

    strategy_names = {
        1: "Hyperparameter Sweep", 2: "Backbone Upgrade",
        3: "Loss Surgery",         4: "Feature-Level KD",
        5: "Data Strategy",        6: "Training Tricks",
    }
    s = proposal.get("strategy", "?")
    print(f"[LLM] Strategy  : {s} — {strategy_names.get(s, 'Unknown')}")
    print(f"[LLM] Proposal  : {proposal.get('description', '')}")
    for patch in proposal.get("patches", []):
        print(f"[LLM]   @ {patch.get('location', '?')}")
        print(f"[LLM]   - {str(patch.get('old',''))[:80]}")
        print(f"[LLM]   + {str(patch.get('new',''))[:80]}")

    applied = llm_helper.apply_proposal(__file__, proposal)
    status  = "APPLIED" if applied else "SKIPPED"
    print(f"[LLM] Status    : {status}")
    print("[LLM] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

    history.append({
        "epoch": epoch, "miou": round(miou, 4),
        "strategy": s, "description": proposal.get("description", ""),
        "patches": proposal.get("patches", []),
        "status": status, "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
    })
    _save_llm_history(history)

# ─────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────

def main():
    print(f"\nDevice : {DEVICE}  |  AMP: {USE_AMP}")
    print(f"Epochs : {EPOCHS}")
    print(f"Samples/epoch: {TOTAL_SAMPLES}  "
          f"(random zoomed/original split each epoch)")

    state     = load_resume()
    start_ep  = state["epoch"] + 1
    best_miou = state["best_miou"]

    # ── Frozen Teacher ─────────────────────────────
    print(f"\n  Loading frozen Teacher from {TEACHER_DIR}")
    proc_t  = Mask2FormerImageProcessor.from_pretrained(
        TEACHER_DIR, do_reduce_labels=False)
    model_t = Mask2FormerForUniversalSegmentation.from_pretrained(
        TEACHER_DIR, num_labels=6,
        ignore_mismatched_sizes=True).to(DEVICE)
    model_t.eval()
    for p in model_t.parameters():
        p.requires_grad = False

    # ── Student — resume from latest epoch ────────
    loaded = False
    for ep in range(start_ep - 1, 0, -1):
        path = os.path.join(OUT_DIR, f"epoch-{ep:03d}")
        if os.path.isdir(path):
            print(f"  Resuming Student from {path}")
            proc_s  = SegformerImageProcessor.from_pretrained(
                path, do_reduce_labels=False)
            model_s = SegformerForSemanticSegmentation.from_pretrained(
                path, num_labels=NUM_CLASSES,
                ignore_mismatched_sizes=True).to(DEVICE)
            loaded = True
            break

    if not loaded:
        print(f"  Loading Student from {STUDENT_DIR}")
        proc_s  = SegformerImageProcessor.from_pretrained(
            STUDENT_DIR, do_reduce_labels=False)
        model_s = SegformerForSemanticSegmentation.from_pretrained(
            STUDENT_DIR, num_labels=NUM_CLASSES,
            ignore_mismatched_sizes=True).to(DEVICE)

    model_s.train()

    # ── Optimizer: differential LR ────────────────
    optimizer = AdamW([
        {"params": model_s.segformer.parameters(),   "lr": LR * 0.1},
        {"params": model_s.decode_head.parameters(), "lr": LR},
    ], weight_decay=0.01)

    # ── KEY FIX: inject initial_lr BEFORE scheduler
    # Required by CosineAnnealingLR when last_epoch > -1
    for pg in optimizer.param_groups:
        pg['initial_lr'] = pg['lr']

    # ── Resumable scheduler ────────────────────────
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
        eta_min=1e-6,
        last_epoch=start_ep - 2 if start_ep > 1 else -1,
    )

    scaler = GradScaler(enabled=USE_AMP)
    cw     = torch.tensor(CE_WEIGHTS, dtype=torch.float32, device=DEVICE)

    # ── EMA ───────────────────────────────────────
    ema = EMA(model_s, decay=EMA_DECAY)

    for epoch in range(start_ep, EPOCHS + 1):

        train_loader, val_loader = build_loaders_for_epoch()
        model_s.train()
        ep_total, ep_kd, ep_ce = [], [], []
        t0 = time.time()

        # ── Manual warmup: scale LR linearly ──────
        if epoch <= WARMUP_EPOCHS:
            warmup_factor = epoch / WARMUP_EPOCHS
            for pg in optimizer.param_groups:
                pg['lr'] = pg['initial_lr'] * warmup_factor

        for step, (imgs, msks) in enumerate(train_loader, 1):
            gt    = labels_tensor(msks, DEVICE)
            s_pix = preprocess(proc_s, imgs, DEVICE)
            t_pix = preprocess(proc_t, imgs, DEVICE)

            with autocast(enabled=USE_AMP):
                s_logits  = model_s(pixel_values=s_pix).logits
                t_logits4 = teacher_logits_4class(model_t, t_pix)

                s_h, s_w = s_logits.shape[-2], s_logits.shape[-1]
                t_small  = F.interpolate(t_logits4, size=(s_h, s_w),
                                         mode="bilinear", align_corners=False)

                # KL-divergence KD (better than MSE)
                s_log = F.log_softmax(s_logits / KD_TEMP, dim=1)
                t_prob = F.softmax(t_small  / KD_TEMP, dim=1)
                kd_loss = F.kl_div(s_log, t_prob,
                                   reduction='batchmean') * (KD_TEMP ** 2)

                s_up    = F.interpolate(s_logits, size=gt.shape[-2:],
                                        mode="bilinear", align_corners=False)
                ce_loss = F.cross_entropy(s_up, gt, weight=cw,
                                          label_smoothing=0.05)

                total = ALPHA * kd_loss + (1.0 - ALPHA) * ce_loss

            optimizer.zero_grad()
            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model_s.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            # EMA update every step
            ema.update(model_s)

            ep_total.append(total.item())
            ep_kd.append(kd_loss.item())
            ep_ce.append(ce_loss.item())

            if step % 20 == 0:
                cur_lr = optimizer.param_groups[-1]['lr']
                print(f"  ep{epoch} step {step:4d} | "
                      f"total={np.mean(ep_total[-20:]):.4f}  "
                      f"kd={np.mean(ep_kd[-20:]):.4f}  "
                      f"ce={np.mean(ep_ce[-20:]):.4f}  "
                      f"lr={cur_lr:.2e}")

        # Only step scheduler after warmup
        if epoch > WARMUP_EPOCHS:
            scheduler.step()

        avg_total = float(np.mean(ep_total))
        elapsed   = time.time() - t0

        # ── Validate student ──────────────────────
        miou, per_class = validate(model_s, proc_s, val_loader)

        # ── Validate EMA (apply → eval → restore) ─
        ema.apply(model_s)
        miou_ema, per_class_ema = validate(model_s, proc_s, val_loader)
        ema.restore(model_s)

        print(f"\nEpoch {epoch}/{EPOCHS} | "
              f"loss={avg_total:.4f}  "
              f"mIoU={miou:.4f}  "
              f"mIoU_ema={miou_ema:.4f}  "
              f"time={elapsed:.0f}s")
        for c in range(NUM_CLASSES):
            print(f"    {CLASS_NAMES[c]:<12}: "
                  f"{per_class[c]:.4f}  (ema: {per_class_ema[c]:.4f})")

        # ── Save epoch checkpoint ─────────────────
        ep_path = os.path.join(OUT_DIR, f"epoch-{epoch:03d}")
        model_s.save_pretrained(ep_path)
        proc_s.save_pretrained(ep_path)

        # ── Save best (student or EMA, whichever wins) ─
        best_this = max(miou, miou_ema)
        if best_this > best_miou:
            best_miou = best_this
            # Apply EMA weights to model, save, then restore
            if miou_ema >= miou:
                ema.apply(model_s)
                model_s.save_pretrained(os.path.join(OUT_DIR, "best"))
                proc_s.save_pretrained(os.path.join(OUT_DIR, "best"))
                ema.restore(model_s)
                print(f"  ★ New best mIoU={best_miou:.4f} (EMA)")
            else:
                model_s.save_pretrained(os.path.join(OUT_DIR, "best"))
                proc_s.save_pretrained(os.path.join(OUT_DIR, "best"))
                print(f"  ★ New best mIoU={best_miou:.4f} (student)")

        state["epoch"]     = epoch
        state["best_miou"] = best_miou
        save_resume(state)

        # ── LLM autotune ──────────────────────────
        run_autotune(epoch, max(miou, miou_ema))
        print("-" * 60)

    print(f"\n  Done. Best mIoU={best_miou:.4f} → {OUT_DIR}/best")


if __name__ == "__main__":
    main()
