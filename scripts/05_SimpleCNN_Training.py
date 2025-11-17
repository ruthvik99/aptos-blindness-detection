#!/usr/bin/env python
# coding: utf-8

# # Simple CNN Baseline

# ## Imports

# In[2]:


import os
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

print("PyTorch:", torch.__version__)
device = torch.device("cpu")
print("Using device:", device)

# ## Load CSV and Image Paths

# In[3]:


train_df = pd.read_csv("../data/raw/train.csv")
image_dir = "../data/raw/train_images/"

train_df.head()

# ## Dataset Class

# In[4]:


from torchvision import transforms

simple_transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor()
])

class SimpleAPTOSDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_id = self.df.loc[idx, 'id_code']
        label  = int(self.df.loc[idx, 'diagnosis'])
        path = os.path.join(self.img_dir, f"{img_id}.png")

        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)

# ## Train/Validation Split

# In[5]:


train_df_, val_df = train_test_split(
    train_df, test_size=0.20, stratify=train_df['diagnosis'], random_state=42
)

train_dataset = SimpleAPTOSDataset(train_df_, image_dir, simple_transform)
val_dataset   = SimpleAPTOSDataset(val_df, image_dir, simple_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False)

len(train_loader), len(val_loader)

# ## SimpleCNN

# In[6]:


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.fc = nn.Sequential(
            nn.Linear(64 * 16 * 16, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 5)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.reshape(x.size(0), -1)
        return self.fc(x)

model = SimpleCNN().to(device)
model

# ## Loss, Optimizer

# In[7]:


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# ## Training Loop

# In[8]:


def train_epoch(model, loader):
    model.train()
    total_loss = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(loader)


def eval_epoch(model, loader):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return total_loss / len(loader), correct / total

# In[9]:


EPOCHS = 5
best_acc = 0

for epoch in range(1, EPOCHS+1):
    train_loss = train_epoch(model, train_loader)
    val_loss, val_acc = eval_epoch(model, val_loader)

    print(f"Epoch {epoch}/{EPOCHS}")
    print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

# ## Evaluation

# In[10]:


y_true, y_pred = [], []

model.eval()
with torch.no_grad():
    for imgs, labels in val_loader:
        outputs = model(imgs.to(device))
        preds = outputs.argmax(dim=1)

        y_true.extend(labels.numpy())
        y_pred.extend(preds.cpu().numpy())

print(classification_report(y_true, y_pred))

# ## Confusion Matrix

# In[11]:


cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(7,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix — SimpleCNN Baseline")
plt.show()
