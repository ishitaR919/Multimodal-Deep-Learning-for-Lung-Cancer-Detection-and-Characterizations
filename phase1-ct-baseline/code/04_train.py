"""Step 4 - fine-tune the ImageNet backbones on the training scans and evaluate on the test scans.


Usage:  python 04_train.py                      # all four models
       python 04_train.py resnet50 vgg16_bn    # selected models


Output (OUT_DIR):        <model>_best.pt, <model>_history.csv, <model>_test_predictions.csv, <model>_test_metrics.csv
Output (FIG_ROOT/per_model): <model>_full_report.png
"""
import os
import sys
import time


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix


from config import PREP_DIR, OUT_DIR, FIG_ROOT, MODELS, BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE
from common import DEVICE, seed_everything, to_model_input, NoduleDataset, build_model, train_one_epoch, evaluate


FIG_DIR = f"{FIG_ROOT}/per_model"
sns.set_style("whitegrid")




def train_model(model_name, train_loader, val_loader, test_loader, class_weights, num_epochs=NUM_EPOCHS):
   print(f"\n{'='*60}\nTraining {model_name} ({num_epochs} epochs)\n{'='*60}")
   model = build_model(model_name)
   criterion = nn.CrossEntropyLoss(weight=class_weights)   # 1:3 imbalance -> keep sensitivity at threshold 0.5
   optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
   scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=3)


   history = []
   best_val_auc = 0.0
   checkpoint_path = f"{OUT_DIR}/{model_name}_best.pt"


   for epoch in range(num_epochs):
       start = time.time()
       train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
       val_metrics = evaluate(model, val_loader)
       scheduler.step(val_metrics["roc_auc"])
       elapsed = time.time() - start


       history.append({
           "epoch": epoch + 1, "train_loss": train_loss, "train_acc": train_acc,
           "val_loss": val_metrics["loss"], "val_acc": val_metrics["accuracy"],
           "val_roc_auc": val_metrics["roc_auc"], "val_pr_auc": val_metrics["pr_auc"],
           "val_sensitivity": val_metrics["recall_sensitivity"], "val_specificity": val_metrics["specificity"],
       })
       print(f"[{model_name}] epoch {epoch+1}/{num_epochs} "
             f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
             f"val_acc={val_metrics['accuracy']:.4f} val_auc={val_metrics['roc_auc']:.4f} ({elapsed:.1f}s)")


       if val_metrics["roc_auc"] > best_val_auc:          # keep the epoch with the best validation ROC-AUC
           best_val_auc = val_metrics["roc_auc"]
           torch.save(model.state_dict(), checkpoint_path)


   history_df = pd.DataFrame(history)
   history_df.to_csv(f"{OUT_DIR}/{model_name}_history.csv", index=False)


   model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
   test_metrics, test_pred_df = evaluate(model, test_loader, return_predictions=True)
   test_pred_df.to_csv(f"{OUT_DIR}/{model_name}_test_predictions.csv", index=False)
   pd.DataFrame([test_metrics]).to_csv(f"{OUT_DIR}/{model_name}_test_metrics.csv", index=False)


   print(f"\n[{model_name}] FINAL TEST METRICS:")
   for k, v in test_metrics.items():
       print(f"  {k}: {v:.4f}")
   return history_df, test_metrics, test_pred_df




def plot_model_report(model_name, history_df, test_metrics, test_pred_df):
   """Train vs val curves, val ROC-AUC/PR-AUC per epoch, and test ROC, PR and confusion matrix."""
   fig, axes = plt.subplots(2, 3, figsize=(16, 9))


   axes[0, 0].plot(history_df["epoch"], history_df["train_loss"], label="train")
   axes[0, 0].plot(history_df["epoch"], history_df["val_loss"], label="val")
   axes[0, 0].set_title(f"{model_name} - Loss"); axes[0, 0].set_xlabel("Epoch"); axes[0, 0].legend()


   axes[0, 1].plot(history_df["epoch"], history_df["train_acc"], label="train")
   axes[0, 1].plot(history_df["epoch"], history_df["val_acc"], label="val")
   axes[0, 1].set_title(f"{model_name} - Accuracy"); axes[0, 1].set_xlabel("Epoch"); axes[0, 1].legend()


   axes[0, 2].plot(history_df["epoch"], history_df["val_roc_auc"], label="val ROC-AUC")
   axes[0, 2].plot(history_df["epoch"], history_df["val_pr_auc"], label="val PR-AUC")
   axes[0, 2].set_title(f"{model_name} - Val ROC-AUC / PR-AUC"); axes[0, 2].set_xlabel("Epoch"); axes[0, 2].legend()


   fpr, tpr, _ = roc_curve(test_pred_df["label"], test_pred_df["prob"])
   axes[1, 0].plot(fpr, tpr, label=f"AUC={test_metrics['roc_auc']:.3f}")
   axes[1, 0].plot([0, 1], [0, 1], "k:", alpha=0.5)
   axes[1, 0].set_title(f"{model_name} - ROC curve (test)")
   axes[1, 0].set_xlabel("False Positive Rate"); axes[1, 0].set_ylabel("True Positive Rate"); axes[1, 0].legend()


   precision, recall, _ = precision_recall_curve(test_pred_df["label"], test_pred_df["prob"])
   axes[1, 1].plot(recall, precision, label=f"AP={test_metrics['pr_auc']:.3f}")
   axes[1, 1].set_title(f"{model_name} - PR curve (test)")
   axes[1, 1].set_xlabel("Recall"); axes[1, 1].set_ylabel("Precision"); axes[1, 1].legend()


   cm = confusion_matrix(test_pred_df["label"], test_pred_df["pred"])
   sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[1, 2], cbar=False)
   axes[1, 2].set_title(f"{model_name} - Confusion Matrix (test)")
   axes[1, 2].set_xlabel("Predicted"); axes[1, 2].set_ylabel("Actual")


   plt.tight_layout()
   save_path = f"{FIG_DIR}/{model_name}_full_report.png"
   plt.savefig(save_path, dpi=150)
   plt.close()
   print(f"Saved {save_path}")




def main(model_names):
   os.makedirs(OUT_DIR, exist_ok=True)
   os.makedirs(FIG_DIR, exist_ok=True)
   print(f"Using device: {DEVICE}")
   seed_everything()


   df = pd.read_csv(f"{PREP_DIR}/labels_split.csv")
   train_df, val_df, test_df = (df[df["split"] == s] for s in ("train", "val", "test"))
   print(f"train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")


   n_pos = int(train_df["label"].sum())
   n_neg = len(train_df) - n_pos
   class_weights = torch.tensor([len(train_df) / (2 * n_neg), len(train_df) / (2 * n_pos)],
                                dtype=torch.float32, device=DEVICE)
   print(f"Class weights (neg, pos): {class_weights.tolist()}")


   train_transform = transforms.Compose([
       transforms.RandomHorizontalFlip(p=0.5),
       transforms.RandomVerticalFlip(p=0.5),
       transforms.RandomRotation(degrees=15),   # empty corners filled with 0 = air
       to_model_input,
   ])
   train_loader = DataLoader(NoduleDataset(train_df, PREP_DIR, train_transform),
                             batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
   val_loader = DataLoader(NoduleDataset(val_df, PREP_DIR, to_model_input),
                           batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
   test_loader = DataLoader(NoduleDataset(test_df, PREP_DIR, to_model_input),
                            batch_size=BATCH_SIZE, shuffle=False, num_workers=2)


   for name in model_names:
       history_df, test_metrics, test_pred_df = train_model(name, train_loader, val_loader, test_loader, class_weights)
       plot_model_report(name, history_df, test_metrics, test_pred_df)




if __name__ == "__main__":
   main(sys.argv[1:] or MODELS)



