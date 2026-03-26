import os
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from torchvision import models, transforms
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

# Paths
BASE_DIR = r"d:\hack_iit tirupati\ii-tt"
MODEL_DIR = os.path.join(BASE_DIR, "Best student zoomed out", "Best student zoomed out")
DATA_DIR = os.path.join(BASE_DIR, "dataset", "train")
IMG_DIR = os.path.join(DATA_DIR, "images")
MASK_OUT_DIR = os.path.join(BASE_DIR, "pred_masks_segformer")
EMBED_CSV = os.path.join(BASE_DIR, "building_roof_embeddings_3class.csv")
EMBED_NPY = os.path.join(BASE_DIR, "building_embeddings_data_3class.npy")
VIS_DIR = os.path.join(BASE_DIR, "roof_cluster_visualizations_3class")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load segmentation model
print('Loading Segformer model...')
seg_model = SegformerForSemanticSegmentation.from_pretrained(MODEL_DIR, local_files_only=True).to(DEVICE)
seg_processor = SegformerImageProcessor.from_pretrained(MODEL_DIR, local_files_only=True)
seg_model.eval()

# Feature extractor model
print('Loading ResNet50 embedding extractor...')
embed_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
embed_model.fc = nn.Identity()
embed_model = embed_model.to(DEVICE)
embed_model.eval()

emb_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

os.makedirs(MASK_OUT_DIR, exist_ok=True)
os.makedirs(VIS_DIR, exist_ok=True)

filenames = sorted([f for f in os.listdir(IMG_DIR) if f.endswith('.npy')])
# For faster initial test, limit to first N files; increase as needed.
filenames = filenames[:10]
print(f'Images to process: {len(filenames)} (first 10 for test)')

records = []
for fname in filenames:
    try:
        img_path = os.path.join(IMG_DIR, fname)
        img_data = np.load(img_path)
    except Exception as e:
        print(f'Error loading {fname}: {e}')
        continue
    if img_data.ndim == 3 and img_data.shape[0] < 10:
        img = img_data[:3, :, :].transpose(1, 2, 0)
    elif img_data.ndim == 3:
        img = img_data[:, :, :3]
    else:
        raise ValueError('Unexpected image shape: ' + str(img_data.shape))

    if img.dtype != np.uint8:
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)

    with torch.no_grad():
        inputs = seg_processor(images=img, return_tensors='pt')
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        outputs = seg_model(**inputs)
        seg_pred = outputs.logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    if seg_pred.shape != img.shape[:2]:
        seg_pred = cv2.resize(seg_pred, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

    np.save(os.path.join(MASK_OUT_DIR, fname), seg_pred)

    building_mask = (seg_pred == 2).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(building_mask, connectivity=8)

    for i in range(1, num_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < 50:
            continue
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])

        crop = img[y:y+h, x:x+w]
        comp_mask = (labels[y:y+h, x:x+w] == i).astype(np.uint8)
        masked_crop = cv2.bitwise_and(crop, crop, mask=comp_mask)

        with torch.no_grad():
            emb_input = emb_transform(masked_crop).unsqueeze(0).to(DEVICE)
            embedding = embed_model(emb_input).cpu().numpy().flatten()

        avg_color = cv2.mean(crop, mask=comp_mask)[:3]

        records.append({
            'tile': fname,
            'building_id': i,
            'x': x, 'y': y, 'w': w, 'h': h, 'area': area,
            'avg_r': avg_color[0], 'avg_g': avg_color[1], 'avg_b': avg_color[2],
            'embedding': embedding
        })

if not records:
    raise RuntimeError('No buildings found in segmentation output to embed')

all_df = pd.DataFrame(records)
emb_matrix = np.vstack(all_df['embedding'].values)
print('Performing KMeans (k=3) on', emb_matrix.shape, 'embeddings')
km = KMeans(n_clusters=3, random_state=42, n_init=10).fit(emb_matrix)
all_df['cluster'] = km.labels_

cluster_color = all_df.groupby('cluster')[['avg_r', 'avg_g', 'avg_b']].mean()
cluster_color['brightness'] = cluster_color[['avg_r', 'avg_g', 'avg_b']].mean(axis=1)
ordered = cluster_color.sort_values('brightness').index.tolist()
cluster_type = {ordered[0]: 'thatched', ordered[1]: 'pucca', ordered[2]: 'tin'}
all_df['roof_type'] = all_df['cluster'].map(cluster_type)

all_df.drop(columns=['embedding']).to_csv(EMBED_CSV, index=False)
np.save(EMBED_NPY, all_df.to_dict('records'))
print('Saved', EMBED_CSV, 'and', EMBED_NPY)

# Optional visualization
import matplotlib.pyplot as plt
import matplotlib.patches as patches
colormap = {0: 'yellow', 1: 'red', 2: 'blue'}
for tile, group in all_df.groupby('tile'):
    img_path = os.path.join(IMG_DIR, tile)
    raw = np.load(img_path)
    if raw.ndim == 3 and raw.shape[0] < 10:
        raw = raw[:3, :, :].transpose(1, 2, 0)
    else:
        raw = raw[:, :, :3]
    if raw.dtype != np.uint8:
        raw = (raw * 255).astype(np.uint8) if raw.max() <= 1.0 else raw.astype(np.uint8)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(raw)
    ax.axis('off')
    for _, r in group.iterrows():
        x, y, w, h, cl = int(r['x']), int(r['y']), int(r['w']), int(r['h']), int(r['cluster'])
        color = colormap.get(cl, 'white')
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        ax.text(x, y-5, f"{cluster_type[cl]}", color='white', fontsize=8, bbox=dict(facecolor=color, alpha=0.6, pad=1))
    outpath = os.path.join(VIS_DIR, f'viz3_{tile.replace(".npy", ".png")}')
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close(fig)

print('Visualizations saved in', VIS_DIR)
