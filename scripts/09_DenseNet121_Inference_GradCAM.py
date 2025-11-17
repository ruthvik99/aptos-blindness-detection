#!/usr/bin/env python
# coding: utf-8

# # DenseNet-121 Inference + Grad-CAM 

# In[6]:


import os
import numpy as np
import pandas as pd
import cv2
import pickle
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# ## Setup

# In[7]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# Reload Dataset Objects (val_loader only)

# load apply_clahe and circular_crop so pickle finds them
def apply_clahe(img, **kwargs):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    return enhanced_img

def circular_crop(img, **kwargs):
    h, w, _ = img.shape
    center = (w//2, h//2)
    r = min(center[0], center[1], w-center[0], h-center[1])
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X-center[0])**2 + (Y-center[1])**2)
    mask = dist <= r
    cropped = np.zeros_like(img)
    cropped[mask] = img[mask]
    return cv2.resize(cropped, (320, 320))

# load APTOSDataset so pickle can load the objects
class APTOSDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_id = self.dataframe.loc[idx, 'id_code']
        label = int(self.dataframe.loc[idx, 'diagnosis'])
        img_path = os.path.join(self.img_dir, f"{img_id}.png")

        image = np.array(Image.open(img_path).convert('RGB'))

        if self.transform:
            image = self.transform(image=image)['image']

        return image, torch.tensor(label, dtype=torch.long)

# Load dataloaders
with open("../data/processed/loaders.pkl", "rb") as f:
    data_objects = pickle.load(f)

val_loader = data_objects["val_loader"]
print("Loaded val_loader:", len(val_loader))

# ## Load Trained DenseNet-121 Checkpoint

# In[13]:


from torchvision.models import densenet121

# 1. Recreate the DenseNet-121 architecture
model = densenet121(weights=None)      # No pretrained weights
model.classifier = nn.Linear(model.classifier.in_features, 5)

model = model.to(device)
model.eval()

# In[15]:


checkpoint_path = "../models/best_densenet121_320.pth"

state_dict = torch.load(checkpoint_path, map_location=device)

# Clean keys if trained under DataParallel
new_state_dict = {}
for k, v in state_dict.items():
    new_state_dict[k.replace("module.", "")] = v

model.load_state_dict(new_state_dict)
print("DenseNet-121 checkpoint loaded successfully!")

# ## Run Inference

# In[ ]:


y_true, y_pred = [], []

with torch.no_grad():
    for imgs, labels in tqdm(val_loader, desc="Running Inference"):
        imgs = imgs.to(device)
        labels = labels.to(device)

        outputs = model(imgs)
        preds = outputs.argmax(dim=1)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

y_true = np.array(y_true)
y_pred = np.array(y_pred)

print("\nClassification Report:\n")
print(classification_report(y_true, y_pred, digits=4))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=[0,1,2,3,4], yticklabels=[0,1,2,3,4])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("DenseNet121 — Confusion Matrix")
plt.show()

# ## Grad-CAM Implementation

# In[17]:


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, target_class=None):
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1)

        loss = output[:, target_class]
        self.model.zero_grad()
        loss.backward(retain_graph=True)

        pooled_grads = torch.mean(self.gradients, dim=[0, 2, 3])
        activations = self.activations[0]

        for i in range(len(pooled_grads)):
            activations[i] *= pooled_grads[i]

        heatmap = torch.mean(activations, dim=0).cpu().numpy()
        heatmap = np.maximum(heatmap, 0)
        heatmap /= heatmap.max()

        return heatmap

# Use last conv layer of DenseNet features
target_layer = model.features[-1]
gradcam = GradCAM(model, target_layer)

print("Grad-CAM Ready")


# ## Grad-CAM Visualization on Sample Images

# In[18]:


val_df = pd.read_csv("../data/raw/train.csv")
from sklearn.model_selection import train_test_split
_, val_df = train_test_split(val_df, test_size=0.2, stratify=val_df['diagnosis'], random_state=42)

sample_ids = val_df.sample(3, random_state=42)["id_code"].values
img_dir = "../data/raw/train_images/"

def show_gradcam(img_path):
    img = Image.open(img_path).convert("RGB")
    img_resized = img.resize((320, 320))

    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],
                             std=[0.229,0.224,0.225])
    ])

    tensor = preprocess(img_resized).unsqueeze(0).to(device)
    heatmap = gradcam.generate(tensor)

    heatmap = cv2.resize(heatmap, (320,320))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(np.array(img_resized), 0.6, heatmap, 0.4, 0)

    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1)
    plt.imshow(img)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1,3,2)
    plt.imshow(heatmap)
    plt.title("Heatmap")
    plt.axis("off")

    plt.subplot(1,3,3)
    plt.imshow(overlay)
    plt.title("Overlay")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

for img_id in sample_ids:
    print(f"🩺 Grad-CAM for Image: {img_id}")
    show_gradcam(os.path.join(img_dir, f"{img_id}.png"))
