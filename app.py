import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
import gradio as gr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# -------------------------------------------
# CONFIG
# -------------------------------------------
CLASSES    = ['crazing','inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']
IMG_SIZE   = 224
MODEL_SAVE = "metal_defect_model.pth"
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------------------
# Load model
# -------------------------------------------
def load_model():
    m = models.efficientnet_b0(weights=None)
    m.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(m.classifier[1].in_features, 256),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(256, len(CLASSES))
    )
    m.load_state_dict(torch.load(MODEL_SAVE, map_location=DEVICE))
    m.to(DEVICE)
    m.eval()
    return m

print("⏳ Loading model...")
model = load_model()
print("✅ Model loaded!")

# -------------------------------------------
# Preprocessing
# -------------------------------------------
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# -------------------------------------------
# GradCAM
# -------------------------------------------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model       = model
        self.gradients   = None
        self.activations = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx):
        self.model.zero_grad()
        output = self.model(input_tensor)
        output[0, class_idx].backward()

        pooled_grads = self.gradients.mean(dim=[0, 2, 3])
        cam = (self.activations[0] * pooled_grads[:, None, None]).sum(dim=0)
        cam = torch.clamp(cam, min=0)
        cam = cam / (cam.max() + 1e-8)
        return cam.cpu().numpy()


# Attach GradCAM to last conv block
gradcam = GradCAM(model, model.features[-1])


def apply_gradcam_overlay(img_rgb, cam):
    cam_resized    = cv2.resize(cam, (img_rgb.shape[1], img_rgb.shape[0]))
    heatmap        = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap        = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    superimposed   = cv2.addWeighted(img_rgb, 0.6, heatmap, 0.4, 0)
    return superimposed

# -------------------------------------------
# Gradio prediction function
# -------------------------------------------
def predict_defect(image):
    # image is numpy RGB from Gradio
    img_tensor = preprocess(image).unsqueeze(0).to(DEVICE)
    img_tensor.requires_grad_()

    with torch.no_grad():
        preds = torch.softmax(model(img_tensor), dim=1)[0]

    pred_idx   = int(preds.argmax())
    pred_class = CLASSES[pred_idx]
    confidence = float(preds[pred_idx]) * 100

    # GradCAM needs grad, re-run with grad
    img_tensor2 = preprocess(image).unsqueeze(0).to(DEVICE)
    cam = gradcam.generate(img_tensor2, pred_idx)

    img_resized   = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    gradcam_img   = apply_gradcam_overlay(img_resized, cam)

    conf_dict = {cls: float(preds[i]) for i, cls in enumerate(CLASSES)}

    return (
        f"🔍 {pred_class}  ({confidence:.1f}% confidence)",
        conf_dict,
        gradcam_img
    )

# -------------------------------------------
# Gradio UI
# -------------------------------------------
with gr.Blocks(title="Metal Defect Detector") as demo:
    gr.Markdown("# 🔩 Metal Surface Defect Detector")
    gr.Markdown("Upload a metal surface image — the model will classify the defect and highlight where it's looking using GradCAM.")

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="numpy", label="Upload Metal Image")
            predict_btn = gr.Button("🔍 Detect Defect", variant="primary")
        with gr.Column():
            result_label  = gr.Textbox(label="Prediction")
            result_conf   = gr.Label(label="Class Probabilities", num_top_classes=6)
            gradcam_image = gr.Image(label="GradCAM — Where the model looks")

    predict_btn.click(
        fn=predict_defect,
        inputs=input_image,
        outputs=[result_label, result_conf, gradcam_image]
    )

print("\n🚀 Launching app at http://127.0.0.1:7860")
demo.launch()