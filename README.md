# 🛰️ Multi-Task Drone Intelligence: Rural Asset Mapping
### AI/ML Hackathon - Ministry of Panchayati Raj (Geospatial Intelligence Challenge)
**Powered by: Geo-Intel Lab, IITTNiF**

---

## 📋 Project Identity
*   **Project Title**: Multi-Task Drone Intelligence: Rooftop Classification, Road & Water Extraction, and Infrastructure Mapping for Rural Panchayats
*   **Team Members**: Ali Rashid, Raghav, Diptesh
*   **Institution**: Vellore Institute of Technology, Vellore
*   **Contact**: ali.rashid2022@vitstudent.ac.in | +91 9555854008
*   **Supported By**: Geospatial Intelligence and Applications Laboratory, IIT Tirupati Navavishkar I-Hub Foundation (IITTNiF)

---

## ⚡ Key Performance Metrics
| Metric | Value |
| :--- | :--- |
| **Overall mIoU** | **83%** (Student Model) |
| **Model Parameters** | **~25M (Edge-Ready)** |
| **Avg Tile Latency** | **0.14s (RTX A4000)** |
| **Training Dataset** | **18,565 labelled tiles** |
| **Village Coverage** | **10 geographically distinct villages** |

---

## 🎯 1. Problem Statement Addressed
Gram Panchayats across rural India manage land and infrastructure for millions, yet operate without accurate, machine-readable geospatial records. Drone surveys under the **SVAMITVA Scheme** generate massive high-resolution imagery, but analysis (building footprints, roads, waterbodies) remains manual, expensive, and slow.

**The Ministry of Panchayati Raj requires four critical deliverables:**
1.  **Building Footprint Extraction** with per-structure rooftop material classification (RCC, Tiled, Tin, Thatched).
2.  **Road Network Delineation** for connectivity assessment (PMGSY).
3.  **Waterbody Mapping** for conservation monitoring (MGNREGS).
4.  **Critical Infrastructure Identification** (Transformers, tanks, wells).

Our end-to-end AI pipeline processes raw drone imagery into structured geospatial outputs in minutes, accelerating rights-of-record (RoR) issuance and local revenue collection.

---

## 🧠 2. Proposed Solution (Three-Module Pipeline)

### 🧩 Module 1 — Semantic Segmentation (Road, Water, Built-up)
*   **Architecture**: Mask2Former (Teacher, Swin-Base) → SegFormer-B2 Student (~25M parameters).
*   **Distillation**: KL-divergence on output probabilities + MSE on intermediate feature maps.
*   **Loss Functions**: Weighted Cross-Entropy + Dice + Focal + **clDice** (Topology-preserving skeletonization loss for road/water continuity).
*   **Performance**: 83% mIoU at 0.14s per tile on RTX A4000.

### 🏠 Module 2 — Rooftop Material Classification (Embedding Pipeline)
*   **Workflow**:
    1.  Segmentation Mask → Connected Components → Building Footprint.
    2.  Crop building → **ResNet-50 Visual Backbone** → 2048-dim Embedding.
    3.  **UMAP/t-SNE** dimensionality reduction for cluster inspection.
    4.  **KMeans Clustering (k=3)**: Discovering Pucca (RCC), Tiled, and Tin/GI Sheet groups automatically.
*   **Innovation**: Unsupervised approach removing the need for manual rooftop annotations.

### 🛰️ Module 3 — Infrastructure Location Detection (Planned Extension)
*   **Target**: Point-scale objects (transformers, tanks, wells) occupying 5–15 pixels.
*   **Strategy**: Separated two-stage architecture using **YOLOv8-nano** or Faster-RCNN with a high-res input head and multi-scale feature pyramids.
*   **Inference**: Tiled inference to maximize recall on small objects.

---

## ✨ 3. Uniqueness and Innovation

1.  **🚀 Rapid Full-Stack Delivery**: Entire pipeline engineered, validated, and documented in just **5 weeks**.
2.  **🏷️ Unsupervised Classification**: Rooftop materials form visually separable clusters automatically via embeddings—eliminating label-collection bottlenecks.
3.  **🤖 LLM-Driven Autotuning**: **Google Gemini 2.0 Flash** embedded in the training loop, autonomously proposing hyperparameter adjustments based on live mIoU trends.
4.  **📉 Edge Optimization**: Knowledge Distillation allows a student model (8× smaller) to retain 83% mIoU, enabling real-time field deployment without cloud dependency.
5.  **🧵 Topology-Preserving Loss**: Incorporation of **clDice loss** to prevent fractured or broken road and waterbody networks.
6.  **📧 Real-Time Monitoring**: Integrated **Resend API** for automated email reports at the end of every epoch (IoU tables + loss breakdowns).

---

## 🛠️ 4. Technology Stack & Methodology

### 📈 4A. Segmentation Pipeline (Module 1)
| Component | Technology | Detail |
| :--- | :--- | :--- |
| **Teacher Model** | Mask2Former (Swin-Base) | State-of-the-art panoptic backbone |
| **Student Model** | SegFormer-B2 | Optimized for edge (~25M params) |
| **Knowledge Distillation** | KL-Div + Feature MSE | Transferring "dark knowledge" to student |
| **Loss Strategy** | CE + Dice + Focal + clDice | Ensuring topological continuity |
| **Monitoring** | Resend API + Gemini 2.0 | Automated reporting and autotuning |

### 🏠 4B. Rooftop Classification Pipeline (Module 2)
1.  **Building Extraction**: 8-connectivity connected components on 'Built-up' mask.
2.  **Visual Embedding**: ResNet-50 (ImageNet) → 2048-dim vector.
3.  **Clustering**: KMeans on embedding space to identify material groups.
4.  **Output**: CSV export with Building ID + Material Label + Visual Overlays.

---

## 📈 5. Expected Impact
| Domain | Estimated Impact |
| :--- | :--- |
| **Property Taxation** | Automatic material classification removes door-to-door enumeration needs. |
| **Infrastructure Planning** | Road networks support PMGSY connectivity gap analysis in minutes. |
| **SVAMITVA Scheme** | Reduces manual digitization effort by an estimated **70–80%**. |
| **Economic Efficiency** | **95% reduction** in per-village processing time (20 mins vs 4 weeks). |

---

## 🗺️ 6. Roadmap

| Phase | Milestone | Timeline |
| :--- | :--- | :--- |
| **Phase 1** | Infrastructure Module (YOLOv8 small-object detector) | Month 1–2 |
| **Phase 2** | Rooftop Classification Refinement (Semi-supervised) | Month 3–4 |
| **Phase 3** | System Integration & Pilot Deployment (3-5 Panchayats) | Month 5–6 |
| **Phase 4** | Accuracy Push (Target ≥95% mIoU) via Field Feedback | Month 7–9 |
| **Phase 5** | State Rollout & NIC Portal Integration | Month 10–12 |

---

## 🏢 7. Stakeholders & Collaborations
*   **Government**: Ministry of Panchayati Raj, NIC, Survey of India, DILRMP/BhuNaksha.
*   **Academic**: Geo-Intel Lab (IIT Tirupati), VIT Vellore.
*   **Industry**: Drone Operators, ESRI India.

---

## 📊 8. Current Stage: Prototype ✓
| Class | IoU Score | Notes |
| :--- | :--- | :--- |
| **Water** | **~90%** | Strong spectral separation from soil/vegetation. |
| **Built-up** | **~86%** | High accuracy on dense rural clusters. |
| **Road** | **~71%** | Improved continuity via clDice despite canopy occlusion. |
| **Overall mIoU** | **83%** | Fully deployable student model. |

---

## 📥 9. Installation & Reproduction
```bash
# Clone & Enter
git clone https://github.com/alirashidAR/ii-tt.git && cd geo-spatial

# Install Dependencies
pip install -r requirements.txt

# Run Inference on TIF
python inference_tif.py --input path/to/drone_ortho.tif --output results/
```

---
*Created for the Ministry of Panchayati Raj AI/ML Hackathon.*
