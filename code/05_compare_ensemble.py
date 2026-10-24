"""Step 5 - compare the trained models and build hard / soft voting ensembles on the test set.


Output (OUT_DIR):            ensemble_predictions.csv, final_metrics_all_models_and_ensembles.csv
Output (FIG_ROOT/comparison): training_curves_all_models.png, roc_curves_comparison.png, pr_curves_comparison.png,
                             confusion_matrices_grid.png, calibration_reliability.png, metric_comparison_bar_chart.png
"""
import os


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
   accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score,
   confusion_matrix, roc_curve, precision_recall_curve, brier_score_loss,
)
from sklearn.calibration import calibration_curve


from config import OUT_DIR, FIG_ROOT, MODELS
from common import expected_calibration_error


FIG_DIR = f"{FIG_ROOT}/comparison"
sns.set_style("whitegrid")




def save(name):
   plt.tight_layout()
   plt.savefig(f"{FIG_DIR}/{name}", dpi=150)
   plt.close()




def main():
   trained_models = [name for name in MODELS
                     if os.path.exists(f"{OUT_DIR}/{name}_history.csv")
                     and os.path.exists(f"{OUT_DIR}/{name}_test_predictions.csv")]
   print(f"Found trained results for: {trained_models}")
   if len(trained_models) < 2:
       print("Need at least 2 trained models to compare/ensemble. Run 04_train.py first.")
       return
   os.makedirs(FIG_DIR, exist_ok=True)
   palette = sns.color_palette("Set2", len(trained_models) + 2)


   # 1. Overlaid training curves (loss / accuracy / val ROC-AUC)
   histories = {name: pd.read_csv(f"{OUT_DIR}/{name}_history.csv") for name in trained_models}
   fig, axes = plt.subplots(1, 3, figsize=(18, 5))
   plot_specs = [("train_loss", "val_loss", "Loss"), ("train_acc", "val_acc", "Accuracy"),
                 (None, "val_roc_auc", "Validation ROC-AUC")]
   for ax, (train_col, val_col, title) in zip(axes, plot_specs):
       for i, (name, hist) in enumerate(histories.items()):
           if train_col:
               ax.plot(hist["epoch"], hist[train_col], "--", color=palette[i], alpha=0.6, label=f"{name} (train)")
           ax.plot(hist["epoch"], hist[val_col], "-", color=palette[i], linewidth=2,
                   label=f"{name} (val)" if train_col else name)
       ax.set_title(title); ax.set_xlabel("Epoch"); ax.legend(fontsize=7)
   save("training_curves_all_models.png")


   # 2. Hard / soft voting ensembles from the saved test predictions
   preds = {name: pd.read_csv(f"{OUT_DIR}/{name}_test_predictions.csv").set_index("filename")
            for name in trained_models}
   base = preds[trained_models[0]][["label"]].copy()
   for name in trained_models:
       base[f"{name}_prob"] = preds[name]["prob"]
       base[f"{name}_pred"] = preds[name]["pred"]


   pred_cols = [f"{name}_pred" for name in trained_models]
   base["hard_vote_prob"] = base[pred_cols].mean(axis=1)
   base["hard_vote_pred"] = (base["hard_vote_prob"] >= 0.5).astype(int)   # 2-2 tie -> positive (favours sensitivity)
   prob_cols = [f"{name}_prob" for name in trained_models]
   base["soft_vote_prob"] = base[prob_cols].mean(axis=1)
   base["soft_vote_pred"] = (base["soft_vote_prob"] >= 0.5).astype(int)
   base.to_csv(f"{OUT_DIR}/ensemble_predictions.csv")
   all_variants = trained_models + ["hard_vote", "soft_vote"]


   # 3. Metrics table for all models + both ensembles
   results = {}
   for name in all_variants:
       y_true, y_pred, y_prob = base["label"], base[f"{name}_pred"], base[f"{name}_prob"]
       tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
       results[name] = {
           "accuracy": accuracy_score(y_true, y_pred),
           "precision": precision_score(y_true, y_pred, zero_division=0),
           "recall_sensitivity": recall_score(y_true, y_pred, zero_division=0),
           "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0.0,
           "f1": f1_score(y_true, y_pred, zero_division=0),
           "roc_auc": roc_auc_score(y_true, y_prob),
           "pr_auc": average_precision_score(y_true, y_prob),
           "brier": brier_score_loss(y_true, y_prob),
           "ece": expected_calibration_error(y_true, y_prob),
       }
   metrics_df = pd.DataFrame(results).T
   metrics_df.to_csv(f"{OUT_DIR}/final_metrics_all_models_and_ensembles.csv")
   print("\n=== Final metrics: individual models + ensembles ===")
   print(metrics_df.round(4))


   # 4. Overlaid ROC curves
   plt.figure(figsize=(7, 6))
   for i, name in enumerate(all_variants):
       fpr, tpr, _ = roc_curve(base["label"], base[f"{name}_prob"])
       style = "--" if "vote" in name else "-"
       plt.plot(fpr, tpr, style, color=palette[i], linewidth=2, label=f"{name} (AUC={results[name]['roc_auc']:.3f})")
   plt.plot([0, 1], [0, 1], "k:", alpha=0.5)
   plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
   plt.title("ROC curves - models vs. voting ensembles"); plt.legend(fontsize=8)
   save("roc_curves_comparison.png")


   # 5. Overlaid PR curves
   plt.figure(figsize=(7, 6))
   for i, name in enumerate(all_variants):
       precision, recall, _ = precision_recall_curve(base["label"], base[f"{name}_prob"])
       style = "--" if "vote" in name else "-"
       plt.plot(recall, precision, style, color=palette[i], linewidth=2, label=f"{name} (AP={results[name]['pr_auc']:.3f})")
   plt.xlabel("Recall"); plt.ylabel("Precision")
   plt.title("Precision-Recall curves - models vs. voting ensembles"); plt.legend(fontsize=8)
   save("pr_curves_comparison.png")


   # 6. Confusion matrix grid
   n, ncols = len(all_variants), 3
   nrows = (n + ncols - 1) // ncols
   fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
   axes = axes.ravel()
   for ax, name in zip(axes, all_variants):
       cm = confusion_matrix(base["label"], base[f"{name}_pred"])
       sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False)
       ax.set_title(name); ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
   for ax in axes[n:]:
       ax.axis("off")
   save("confusion_matrices_grid.png")


   # 7. Reliability diagram (calibration)
   plt.figure(figsize=(7, 6))
   for i, name in enumerate(all_variants):
       if name == "hard_vote":          # vote fractions take only 5 values; not a calibrated probability
           continue
       frac_pos, mean_pred = calibration_curve(base["label"], base[f"{name}_prob"], n_bins=10, strategy="quantile")
       plt.plot(mean_pred, frac_pos, "o-", color=palette[i], label=f"{name} (ECE={results[name]['ece']:.3f})")
   plt.plot([0, 1], [0, 1], "k:", alpha=0.5)
   plt.xlabel("Mean predicted probability"); plt.ylabel("Observed fraction of nodules")
   plt.title("Reliability diagram (test set)"); plt.legend(fontsize=8)
   save("calibration_reliability.png")


   # 8. Metric bar chart
   plot_metrics = ["accuracy", "precision", "recall_sensitivity", "specificity", "f1", "roc_auc", "pr_auc"]
   melted = (metrics_df[plot_metrics].reset_index().rename(columns={"index": "model"})
             .melt(id_vars="model", var_name="metric", value_name="score"))
   plt.figure(figsize=(14, 6))
   sns.barplot(data=melted, x="metric", y="score", hue="model")
   plt.title("Model-wise and ensemble-wise metric comparison"); plt.ylim(0, 1.05)
   plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
   save("metric_comparison_bar_chart.png")


   print(f"\nAll comparison figures saved to {FIG_DIR}")


if __name__ == "__main__":
   main()

