Here’s a cleaner, sharper, and more professional version of your README. I’ve tightened language, improved structure, and removed redundancy while keeping your technical depth intact.

---

# 🛰️ Multi-Task Drone Intelligence: Rural Asset Mapping

### AI/ML Hackathon — Ministry of Panchayati Raj (Geospatial Intelligence Challenge)

**Powered by: Geo-Intel Lab, IITTNiF**

---

## 📋 Project Overview

**Title**: Multi-Task Drone Intelligence for Rural Panchayats
**Scope**: Rooftop Classification, Road & Water Extraction, Infrastructure Mapping

**Team**: Ali Rashid, Raghav, Diptesh
**Institution**: Vellore Institute of Technology, Vellore
**Contact**: [ali.rashid2022@vitstudent.ac.in](mailto:ali.rashid2022@vitstudent.ac.in) | +91 9555854008
**Supported By**: Geospatial Intelligence and Applications Laboratory, IIT Tirupati Navavishkar I-Hub Foundation (IITTNiF)

---

## ⚡ Key Metrics

| Metric                  | Value                            |
| :---------------------- | :------------------------------- |
| **Overall mIoU**        | **83% (Student Model)**          |
| **Model Size**          | **~25M parameters (Edge-ready)** |
| **Inference Latency**   | **0.14s per tile (RTX A4000)**   |
| **Dataset Size**        | **18,565 labeled tiles**         |
| **Geographic Coverage** | **10 villages**                  |

---

## 🎯 Problem Statement

Gram Panchayats across rural India lack accurate, machine-readable geospatial records. While the **SVAMITVA Scheme** generates high-resolution drone imagery, extracting actionable insights such as building footprints, roads, and waterbodies remains manual, slow, and costly.

### Required Outputs

1. Building footprint extraction with rooftop material classification (RCC, Tiled, Tin, Thatched)
2. Road network delineation for connectivity analysis (PMGSY)
3. Waterbody mapping for conservation (MGNREGS)
4. Critical infrastructure detection (transformers, tanks, wells)

### Our Contribution

An end-to-end AI pipeline that converts raw drone imagery into structured geospatial outputs in minutes, enabling faster land record generation and improved governance workflows.

---

## 🧠 System Architecture

### 🧩 Module 1 — Semantic Segmentation

**Classes**: Road, Water, Built-up

* **Teacher Model**: Mask2Former (Swin-Base)
* **Student Model**: SegFormer-B2 (~25M parameters)
* **Distillation**: KL divergence + feature-level MSE
* **Loss**: Cross-Entropy + Dice + Focal + **clDice**
* **Performance**: 83% mIoU at 0.14s/tile

---

### 🏠 Module 2 — Rooftop Material Classification

**Pipeline**:

1. Segmentation → connected components → building footprints
2. Cropping → ResNet-50 → 2048-d embeddings
3. Dimensionality reduction (UMAP / t-SNE)
4. KMeans clustering (k=3)

**Key Insight**:
Unsupervised clustering enables material classification without manual labels.

---

### 🛰️ Module 3 — Infrastructure Detection (Planned)

* **Targets**: Small objects (5–15 pixels) such as transformers and wells
* **Approach**: YOLOv8-nano or Faster R-CNN with multi-scale feature pyramids
* **Inference Strategy**: High-resolution tiled detection

---

## ✨ Key Innovations

* **Rapid Development**: Full pipeline designed and validated in 5 weeks
* **Unsupervised Learning**: Rooftop classification without annotation overhead
* **LLM-Assisted Training**: Google Gemini 2.0 Flash used for adaptive hyperparameter tuning
* **Edge Optimization**: 8× model compression with minimal performance loss
* **Topology-Aware Learning**: clDice loss ensures continuous road and water structures
* **Automated Monitoring**: Epoch-level reporting via Resend API

---

## 🛠️ Technology Stack

### Segmentation Pipeline

| Component    | Technology                 | Purpose                          |
| :----------- | :------------------------- | :------------------------------- |
| Teacher      | Mask2Former (Swin-Base)    | High-quality supervision         |
| Student      | SegFormer-B2               | Edge deployment                  |
| Distillation | KL + MSE                   | Knowledge transfer               |
| Loss         | CE + Dice + Focal + clDice | Accuracy + topology preservation |
| Monitoring   | Resend API + Gemini        | Reporting and autotuning         |

---

### Rooftop Classification Pipeline

* Connected component analysis (8-connectivity)
* ResNet-50 embeddings (ImageNet pretrained)
* KMeans clustering for material grouping
* CSV outputs with building-level metadata

---

## 📈 Impact

| Domain                      | Impact                            |
| :-------------------------- | :-------------------------------- |
| **Property Taxation**       | Eliminates manual surveys         |
| **Infrastructure Planning** | Rapid connectivity analysis       |
| **SVAMITVA Scheme**         | 70–80% reduction in manual effort |
| **Processing Efficiency**   | 20 minutes vs 4 weeks per village |

---

## 🗺️ Roadmap

| Phase   | Milestone                             | Timeline    |
| :------ | :------------------------------------ | :---------- |
| Phase 1 | Infrastructure detection module       | Month 1–2   |
| Phase 2 | Semi-supervised rooftop refinement    | Month 3–4   |
| Phase 3 | Pilot deployment (3–5 Panchayats)     | Month 5–6   |
| Phase 4 | Accuracy optimization (≥95% mIoU)     | Month 7–9   |
| Phase 5 | State-scale rollout & NIC integration | Month 10–12 |

---

## 🏢 Stakeholders

* **Government**: Ministry of Panchayati Raj, NIC, Survey of India, DILRMP
* **Academic**: Geo-Intel Lab (IIT Tirupati), VIT Vellore
* **Industry**: Drone operators, ESRI India

---

## 📊 Current Performance

| Class       | IoU     | Notes                          |
| :---------- | :------ | :----------------------------- |
| Water       | ~90%    | Strong spectral separability   |
| Built-up    | ~86%    | Robust in dense regions        |
| Road        | ~71%    | Improved continuity via clDice |
| **Overall** | **83%** | Production-ready               |

---

## 📥 Installation & Usage

```bash
# Clone repository
git clone https://github.com/alirashidAR/ii-tt.git
cd geo-spatial

# Install dependencies
pip install -r requirements.txt

# Run inference
python inference_tif.py --input path/to/drone_ortho.tif --output results/
```

### Models & Results

[https://drive.google.com/drive/folders/11RXtWffYmsXESZhzA5n3Glw-fPToShr1?usp=sharing](https://drive.google.com/drive/folders/11RXtWffYmsXESZhzA5n3Glw-fPToShr1?usp=sharing)

---

## 📌 Status

**Stage**: Prototype complete and deployable

---

*Developed for the Ministry of Panchayati Raj AI/ML Hackathon.*
