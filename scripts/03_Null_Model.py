#!/usr/bin/env python
# coding: utf-8

# # Null Model Baseline

# In[1]:


import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 

# # Load dataset

# In[2]:


df = pd.read_csv("../data/raw/train.csv")
df.head()

# ## Class distribution

# In[3]:


class_counts = df['diagnosis'].value_counts()
print("Class Distribution:\n", class_counts)

# ## Identify majority class

# In[4]:


majority_class = class_counts.idxmax()
print(f"\nMajority Class = {majority_class}")

# ## Split data

# In[5]:


from sklearn.model_selection import train_test_split
train_df, val_df = train_test_split(
    df, test_size=0.2, stratify=df['diagnosis'], random_state=42
)

y_true = val_df['diagnosis'].values
y_pred = np.full_like(y_true, fill_value=majority_class)

# ## Metrics

# In[6]:


print("\nAccuracy:", accuracy_score(y_true, y_pred))
print("\nClassification Report:\n", classification_report(y_true, y_pred))

# ## Confusion matrix

# In[7]:


cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=np.unique(y_true),
            yticklabels=np.unique(y_true))
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Null Model — Confusion Matrix")
plt.show()
