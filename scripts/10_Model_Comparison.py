#!/usr/bin/env python
# coding: utf-8

# # Model Comparison

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

# In[2]:


results = {
    "Null Model": {
        "Accuracy": 0.4925,
        "Precision": 0.24,
        "Recall": 0.49,
        "F1": 0.33,
        "Params": 0,
        "Image Size": "N/A",
        "Augmentation": "None",
        "Training Time (min)": 0
    },
    
    "Logistic Regression": {
        "Accuracy": 0.6317,
        "Precision": 0.6779,
        "Recall": 0.6317,
        "F1": 0.6449,
        "Params": 11,  # 11 features
        "Image Size": "Raw (~ varying)",
        "Augmentation": "None",
        "Training Time (min)": 0.1
    },
    
    "SimpleCNN": {
        "Accuracy": 0.71,
        "Precision": 0.59,
        "Recall": 0.71,
        "F1": 0.63,
        "Params": 2_121_509,
        "Image Size": "128×128",
        "Augmentation": "None (raw)",
        "Training Time (min)": 24.9
    },
    
    "ResNet-50": {
        "Accuracy": 0.7271,
        "Precision": 0.7633,
        "Recall": 0.7271,
        "F1": 0.7342,
        "Params": 25_557_032,
        "Image Size": "224×224",
        "Augmentation": "CLAHE + CircularCrop + Resize + Rotate + Bright/Contrast + Blur",
        "Training Time (min)": 57.3
    },
    
    "DenseNet-121": {
        "Accuracy": 0.7735,
        "Precision": 0.7814,
        "Recall": 0.7735,
        "F1": 0.7744,
        "Params": 7_978_856,
        "Image Size": "320×320",
        "Augmentation": "CLAHE + CircularCrop + Resize + Rotate + Flip + Bright/Contrast + Blur",
        "Training Time (min)": 50.5
    }
}

# In[3]:


df = pd.DataFrame(results).T
df

# In[7]:


plt.figure(figsize=(10,6))
sns.barplot(x=df.index, y=df["Accuracy"], palette="viridis")
plt.xticks(rotation=45)
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.xlabel("Model")
plt.ylim(0, 1)
plt.show()

# In[8]:


plt.figure(figsize=(10,6))
sns.barplot(x=df.index, y=df["F1"], palette="magma")
plt.xticks(rotation=45)
plt.title("Model Macro F1 Score Comparison")
plt.ylabel("Macro F1 Score")
plt.ylim(0, 1)
plt.show()

# In[9]:


plt.figure(figsize=(10,6))
sns.barplot(x=df.index, y=df["Params"], palette="crest")
plt.yscale("log")
plt.title("Model Parameters (log scale)")
plt.ylabel("Number of Parameters (log)")
plt.xticks(rotation=45)
plt.show()

# In[11]:


plt.figure(figsize=(10,6))
sns.barplot(x=df.index, y=df["Training Time (min)"], palette="rocket")
plt.xticks(rotation=45)
plt.title("Training Time Comparison")
plt.ylabel("Minutes")
plt.show()

# In[12]:


df[["Image Size", "Augmentation", "Params", "Training Time (min)"]]

# In[13]:


best_model = df["Accuracy"].idxmax()
print(f"Best model based on validation accuracy: **{best_model}**")
