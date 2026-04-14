# 🔩 Metal Defect Detection System

An end-to-end deep learning system that automatically detects and classifies 
surface defects in metal components — built to automate industrial quality 
control inspection using computer vision.

> Trained on the NEU Metal Surface Defects benchmark dataset — 1,800 images across 6 defect classes.

<img width="1607" height="913" alt="Screenshot 2026-04-08 185018" src="https://github.com/user-attachments/assets/6145ca0b-6241-473a-a5cb-098fe91deba7" />

---

## 📊 Results

| Metric | Score |
|--------|-------|
| **Final Test Accuracy** | **99%** |
| Macro Avg Precision | 0.99 |
| Macro Avg Recall | 0.99 |
| Macro Avg F1-Score | 0.99 |
| Classes | 6 defect types |
| Training Images | 1,440 |
| Validation Images | 360 |

---

## 📸 Training Results

![Training History](training_history.png)

---

## 🎯 Defect Classes & Per-Class Performance

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Crazing | 1.00 | 1.00 | 1.00 |
| Inclusion | 1.00 | 0.93 | 0.97 |
| Patches | 1.00 | 1.00 | 1.00 |
| Pitted Surface | 1.00 | 1.00 | 1.00 |
| Rolled-in Scale | 1.00 | 1.00 | 1.00 |
| Scratches | 0.94 | 1.00 | 0.97 |

---

## 🧠 Model Architecture & Training Strategy

- **Base:** Pre-trained CNN backbone (transfer learning)
- **Phase 1:** Classification head trained for 10 epochs (frozen backbone)
- **Phase 2:** Full fine-tuning of top layers for 10 epochs
- **Output:** Predicted defect class + confidence score
- **Saved Model:** `metal_defect_model.pth`

This two-phase transfer learning approach allowed the model to first learn
task-specific features before fine-tuning the backbone — resulting in stable
convergence and 99% test accuracy.

---

## 🛠 Tech Stack

- **Framework:** PyTorch + torchvision
- **Vision:** OpenCV
- **Data & Metrics:** scikit-learn, NumPy, Matplotlib
- **Dataset:** [NEU Metal Surface Defects Dataset (Kaggle)](https://www.kaggle.com/datasets/fantacher/neu-metal-surface-defects-data)

---

## 🚀 Getting Started

**1. Clone the repo**
```bash
git clone https://github.com/keertiG-1296/metal-detection-system.git
cd metal-detection-system
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run inference**
```bash
python app.py
```

**4. Train from scratch**
```bash
python train.py
```