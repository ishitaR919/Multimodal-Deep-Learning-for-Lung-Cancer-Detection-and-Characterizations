"""Model, dataset and evaluation code shared by training, ensembling and Grad-CAM."""
import os
import random


import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import models, transforms
from sklearn.metrics import (
   accuracy_score, precision_score, recall_score, f1_score,
   roc_auc_score, average_precision_score, confusion_matrix, brier_score_loss,
)


from config import INPUT_SIZE, IMAGENET_MEAN, IMAGENET_STD, SEED


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")




def seed_everything(seed=SEED):
   random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)




# applied to every split, after augmentation: upsample + ImageNet normalisation (matches pretraining)
to_model_input = transforms.Compose([
   transforms.Resize((INPUT_SIZE, INPUT_SIZE), antialias=True),
   transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])




def expected_calibration_error(y, p, n_bins=10):
   """ECE (Guo et al. 2017): bin-size-weighted gap between mean confidence and observed positive rate."""
   y, p = np.asarray(y), np.asarray(p)
   bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
   return float(sum(abs(y[bins == b].mean() - p[bins == b].mean()) * (bins == b).mean()
                    for b in range(n_bins) if (bins == b).any()))




class NoduleDataset(Dataset):
   def __init__(self, df, data_dir, transform=None):
       self.df = df.reset_index(drop=True)
       self.data_dir = data_dir
       self.transform = transform


   def __len__(self):
       return len(self.df)


   def __getitem__(self, idx):
       row = self.df.iloc[idx]
       crop = np.load(os.path.join(self.data_dir, row["filename"]))   # (3, H, W): axial, coronal, sagittal
       crop_tensor = torch.from_numpy(crop).float()
       if self.transform:
           crop_tensor = self.transform(crop_tensor)
       return crop_tensor, int(row["label"]), row["filename"]




def build_model(name):
   """ImageNet-pretrained backbone with a new 2-class output layer."""
   if name == "resnet50":
       m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
       m.fc = nn.Linear(m.fc.in_features, 2)
   elif name == "densenet121":
       m = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
       m.classifier = nn.Linear(m.classifier.in_features, 2)
   elif name == "efficientnet_b0":
       m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
       m.classifier[1] = nn.Linear(m.classifier[1].in_features, 2)
   elif name == "vgg16_bn":
       m = models.vgg16_bn(weights=models.VGG16_BN_Weights.IMAGENET1K_V1)
       m.classifier[6] = nn.Linear(m.classifier[6].in_features, 2)
   else:
       raise ValueError(f"Unknown model name: {name}")
   return m.to(DEVICE)




def train_one_epoch(model, loader, optimizer, criterion):
   model.train()
   running_loss, correct, total = 0.0, 0, 0
   for images, labels, _ in loader:
       images, labels = images.to(DEVICE), labels.to(DEVICE)
       optimizer.zero_grad()
       outputs = model(images)
       loss = criterion(outputs, labels)
       loss.backward()
       optimizer.step()
       running_loss += loss.item() * images.size(0)
       correct += (outputs.argmax(1) == labels).sum().item()
       total += images.size(0)
   return running_loss / total, correct / total




@torch.no_grad()
def evaluate(model, loader, return_predictions=False):
   model.eval()
   all_labels, all_preds, all_probs, all_files = [], [], [], []
   running_loss, criterion = 0.0, nn.CrossEntropyLoss()


   for images, labels, filenames in loader:
       images_dev, labels_dev = images.to(DEVICE), labels.to(DEVICE)
       outputs = model(images_dev)
       loss = criterion(outputs, labels_dev)
       running_loss += loss.item() * images.size(0)


       probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
       preds = outputs.argmax(dim=1).cpu().numpy()


       all_labels.extend(labels.numpy())
       all_preds.extend(preds)
       all_probs.extend(probs)
       all_files.extend(filenames)


   metrics = {
       "loss": running_loss / len(loader.dataset),
       "accuracy": accuracy_score(all_labels, all_preds),
       "precision": precision_score(all_labels, all_preds, zero_division=0),
       "recall_sensitivity": recall_score(all_labels, all_preds, zero_division=0),
       "f1": f1_score(all_labels, all_preds, zero_division=0),
       "roc_auc": roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else float("nan"),
       "pr_auc": average_precision_score(all_labels, all_probs),
       "brier": brier_score_loss(all_labels, all_probs),
       "ece": expected_calibration_error(all_labels, all_probs),
   }
   tn, fp, fn, tp = confusion_matrix(all_labels, all_preds, labels=[0, 1]).ravel()
   metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0.0


   if return_predictions:
       pred_df = pd.DataFrame({
           "filename": all_files, "label": all_labels,
           "pred": all_preds, "prob": all_probs,
       })
       return metrics, pred_df
   return metrics
