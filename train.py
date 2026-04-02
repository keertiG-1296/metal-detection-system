import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# -------------------------------------------
# CONFIG
# -------------------------------------------
DATASET_PATH  = r"C:\Users\HP\Downloads\archive (1)\NEU-DET"
TRAIN_PATH    = os.path.join(DATASET_PATH, "train", "images")
VAL_PATH      = os.path.join(DATASET_PATH, "validation", "images")
CLASSES       = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']
IMG_SIZE      = 224
BATCH_SIZE    = 32
EPOCHS_PHASE1 = 10
EPOCHS_PHASE2 = 10
MODEL_SAVE    = "metal_defect_model.pth"
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f" Using device: {DEVICE}")

# -------------------------------------------
# STEP 1 — Dataset check
# -------------------------------------------
for split, path in [("TRAIN", TRAIN_PATH), ("VALIDATION", VAL_PATH)]:
    print(f"\n{'='*40}\n{split}\n{'='*40}")
    for cls in CLASSES:
        folder = os.path.join(path, cls)
        print(f"  {cls} → {len(os.listdir(folder))} images")

# -------------------------------------------
# STEP 2 — Visualize one sample per class
# -------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
fig.suptitle("NEU Metal Surface Defects — One Sample Per Class", fontsize=14)
for idx, cls in enumerate(CLASSES):
    folder   = os.path.join(TRAIN_PATH, cls)
    img_path = os.path.join(folder, os.listdir(folder)[0])
    img      = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    row, col = idx // 3, idx % 3
    axes[row, col].imshow(img, cmap='gray')
    axes[row, col].set_title(cls, fontsize=12)
    axes[row, col].axis('off')
plt.tight_layout()
plt.show()

# -------------------------------------------
# STEP 3 — Data loaders
# -------------------------------------------
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(TRAIN_PATH, transform=train_transform)
val_dataset   = datasets.ImageFolder(VAL_PATH,   transform=val_transform)

train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
val_loader    = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"\n Class mapping: {train_dataset.class_to_idx}")

# -------------------------------------------
# STEP 4 — Build model (EfficientNetB0)
# -------------------------------------------
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

# Freeze all layers initially
for param in model.parameters():
    param.requires_grad = False

# Replace classifier head
model.classifier = nn.Sequential(
    nn.Dropout(p=0.4),
    nn.Linear(model.classifier[1].in_features, 256),
    nn.ReLU(),
    nn.Dropout(p=0.3),
    nn.Linear(256, len(CLASSES))
)
model = model.to(DEVICE)

# -------------------------------------------
# STEP 5 — Training helper functions
# -------------------------------------------
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        correct    += (outputs.argmax(1) == labels).sum().item()
        total      += images.size(0)
    return total_loss / total, correct / total


def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs    = model(images)
            loss       = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            correct    += (outputs.argmax(1) == labels).sum().item()
            total      += images.size(0)
    return total_loss / total, correct / total


def run_training(model, loader, val_loader, optimizer, criterion, epochs, phase):
    history = {'loss': [], 'acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0
    patience_counter = 0
    PATIENCE = 5

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc   = train_one_epoch(model, loader, optimizer, criterion)
        val_loss, val_acc = evaluate(model, val_loader, criterion)

        history['loss'].append(tr_loss)
        history['acc'].append(tr_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"[{phase}] Epoch {epoch}/{epochs} — "
              f"Loss: {tr_loss:.4f} | Acc: {tr_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_SAVE)  # Save best
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f" Early stopping at epoch {epoch}")
                break

    return history

# -------------------------------------------
# STEP 6 — Phase 1: Train head only
# -------------------------------------------
criterion  = nn.CrossEntropyLoss()
optimizer1 = optim.Adam(model.classifier.parameters(), lr=1e-3)

print("\n Phase 1: Training classification head...")
history1 = run_training(model, train_loader, val_loader,
                        optimizer1, criterion, EPOCHS_PHASE1, "Phase 1")

# -------------------------------------------
# STEP 7 — Phase 2: Fine-tune last 3 blocks
# -------------------------------------------
print("\n Phase 2: Fine-tuning top layers...")
for name, param in model.named_parameters():
    if "features.7" in name or "features.8" in name or "classifier" in name:
        param.requires_grad = True

optimizer2 = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4
)

history2 = run_training(model, train_loader, val_loader,
                        optimizer2, criterion, EPOCHS_PHASE2, "Phase 2")

# -------------------------------------------
# STEP 8 — Plot training history
# -------------------------------------------
def plot_history(h1, h2):
    acc      = h1['acc']      + h2['acc']
    val_acc  = h1['val_acc']  + h2['val_acc']
    loss     = h1['loss']     + h2['loss']
    val_loss = h1['val_loss'] + h2['val_loss']
    epochs   = range(1, len(acc) + 1)
    split    = len(h1['acc'])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, acc,    'b-o', label='Train Acc')
    ax1.plot(epochs, val_acc,'r-o', label='Val Acc')
    ax1.axvline(x=split, color='gray', linestyle='--', label='Fine-tune start')
    ax1.set_title('Accuracy'); ax1.legend(); ax1.grid(True)

    ax2.plot(epochs, loss,    'b-o', label='Train Loss')
    ax2.plot(epochs, val_loss,'r-o', label='Val Loss')
    ax2.axvline(x=split, color='gray', linestyle='--', label='Fine-tune start')
    ax2.set_title('Loss'); ax2.legend(); ax2.grid(True)

    plt.suptitle("Training History — Phase 1 + Phase 2", fontsize=14)
    plt.tight_layout()
    plt.savefig("training_history.png", dpi=150)
    plt.show()
    print("Saved training_history.png")

plot_history(history1, history2)

# -------------------------------------------
# STEP 9 — Evaluate with classification report
# -------------------------------------------
print("\n Final Evaluation...")
model.load_state_dict(torch.load(MODEL_SAVE))  # Load best checkpoint
model.eval()

all_preds, all_labels = [], []
with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(DEVICE)
        preds  = model(images).argmax(1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=CLASSES))

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASSES, yticklabels=CLASSES)
plt.title("Confusion Matrix")
plt.ylabel("Actual"); plt.xlabel("Predicted")
plt.tight_layout(); plt.show()

print(f"\n Model saved to: {MODEL_SAVE}")