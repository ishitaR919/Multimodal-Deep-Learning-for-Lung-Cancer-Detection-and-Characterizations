"""Step 8 - full evaluation of every model and ensemble on the pooled out-of-fold patient predictions:
results table with patient-bootstrap 95% CIs, ROC / PR curves, CI bars, per-fold spread, confusion matrices,
calibration, training curves and an ablation view.


Usage:  python 08_evaluate.py


Input:  oof_*.csv, history_*.csv (steps 5-7)
Output (RES_DIR): results_all_models.csv
Output (FIG_DIR): eval1_roc_pr.png, eval2_auc_bars_ci.png, eval3_per_fold_auc.png, eval4_confusion_matrices.png,
                 eval5_calibration.png, eval6_training_curves.png
"""
import os


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve


from config import RES_DIR, FIG_DIR, MODEL_SPECS
from common import load_state, load_oof, summarize_oof, per_fold_auc


sns.set_style("whitegrid")
ALL = ["tab_clinical_lr", "tab_radiomics_lr", "tab_clin_radiomics_lr", "tab_clin_radiomics_rf",
      "ct_only", "pet_only", "early", "intermediate", "intermediate_clin", "cross_attn", "cross_attn_md",
      "cross_attn_md_clin", "late_ct_pet", "late_ct_pet_clin", "ens_soft_fusion", "ens_hard_fusion",
      "ens_soft_image_all", "ens_hard_image_all"]




def main():
   st = load_state(with_crops=False)
   oof = load_oof(ALL)
   hist = {m: pd.read_csv(f"{RES_DIR}/history_{m}.csv") for m in MODEL_SPECS if os.path.exists(f"{RES_DIR}/history_{m}.csv")}
   ORDER = [m for m in ["tab_clinical_lr", "tab_radiomics_lr", "tab_clin_radiomics_lr", "tab_clin_radiomics_rf",
                        "ct_only", "pet_only", "early", "intermediate", "intermediate_clin", "cross_attn", "cross_attn_md",
                        "cross_attn_md_clin", "late_ct_pet", "late_ct_pet_clin", "ens_soft_fusion", "ens_hard_fusion",
                        "ens_soft_image_all", "ens_hard_image_all"] if m in oof]
   RES = summarize_oof({m: oof[m] for m in ORDER}, B=1000)
   RES.to_csv(f"{RES_DIR}/results_all_models.csv")
   show = RES[["roc_auc", "roc_auc_lo", "roc_auc_hi", "pr_auc", "balanced_acc", "sensitivity", "specificity", "f1", "brier"]].astype(float).round(3)
   print("=== Pooled out-of-fold results (patient level; threshold 0.5; 95% CI by patient bootstrap) ===")
   print(show.to_string())
   prev = float(np.mean([st.labels[p] for p in st.pids]))
   print(f"\nPositive-class prevalence = {prev:.3f}  (PR-AUC of a random classifier ~ {prev:.3f})")


   GROUP_A = [m for m in ["tab_clin_radiomics_lr", "ct_only", "pet_only", "early"] if m in oof]
   GROUP_B = [m for m in ["intermediate", "intermediate_clin", "cross_attn", "cross_attn_md", "cross_attn_md_clin",
                          "late_ct_pet", "ens_soft_fusion", "ens_hard_fusion"] if m in oof]
   pal = sns.color_palette("tab10", 12)


   fig, axes = plt.subplots(2, 2, figsize=(14, 12))
   for col, grp in enumerate((GROUP_A, GROUP_B)):
       for i, m in enumerate(grp):
           d = oof[m]
           fpr, tpr, _ = roc_curve(d["label"], d["prob"]); axes[0, col].plot(fpr, tpr, lw=2, color=pal[i], label=f"{m} ({RES.loc[m, 'roc_auc']:.2f})")
           pr, rc, _ = precision_recall_curve(d["label"], d["prob"]); axes[1, col].plot(rc, pr, lw=2, color=pal[i], label=f"{m} ({RES.loc[m, 'pr_auc']:.2f})")
       axes[0, col].plot([0, 1], [0, 1], "k:"); axes[0, col].set_title("ROC (pooled out-of-fold)"); axes[0, col].set_xlabel("1 - specificity"); axes[0, col].set_ylabel("sensitivity")
       axes[1, col].axhline(prev, color="k", ls=":"); axes[1, col].set_title("Precision-Recall (pooled out-of-fold)"); axes[1, col].set_xlabel("recall"); axes[1, col].set_ylabel("precision")
       for r in (0, 1):
           axes[r, col].legend(fontsize=8)
   plt.tight_layout(); plt.savefig(f"{FIG_DIR}/eval1_roc_pr.png", dpi=140); plt.close()


   fig, axes = plt.subplots(1, 2, figsize=(15, max(5, 0.38 * len(ORDER))))
   for ax, (metric, lo, hi, base) in zip(axes, [("roc_auc", "roc_auc_lo", "roc_auc_hi", 0.5), ("pr_auc", "pr_auc_lo", "pr_auc_hi", prev)]):
       r = RES.loc[ORDER, [metric, lo, hi]].astype(float)
       y = np.arange(len(r))
       ax.barh(y, r[metric], xerr=[r[metric] - r[lo], r[hi] - r[metric]], color=["#8172B3" if m.startswith(("ens", "late")) else "#4C72B0" if m.startswith("tab") else "#55A868" for m in r.index], capsize=3)
       ax.set_yticks(y); ax.set_yticklabels(r.index); ax.invert_yaxis(); ax.axvline(base, color="k", ls=":"); ax.set_title(f"{metric} with 95% CI")
   plt.tight_layout(); plt.savefig(f"{FIG_DIR}/eval2_auc_bars_ci.png", dpi=140); plt.close()


   pf = pd.DataFrame({m: per_fold_auc(oof[m]) for m in ORDER}).T
   plt.figure(figsize=(11, max(4, 0.35 * len(ORDER))))
   for i, m in enumerate(pf.index):
       plt.scatter(pf.loc[m].values, [i] * pf.shape[1], color="#4C72B0", s=40, alpha=0.8)
       plt.plot([np.nanmin(pf.loc[m]), np.nanmax(pf.loc[m])], [i, i], color="#4C72B0", alpha=0.4)
   plt.yticks(range(len(pf)), pf.index); plt.gca().invert_yaxis(); plt.axvline(0.5, color="k", ls=":"); plt.xlabel("ROC-AUC per fold")
   plt.title("Fold-to-fold variability (each dot = one test fold)"); plt.tight_layout(); plt.savefig(f"{FIG_DIR}/eval3_per_fold_auc.png", dpi=140); plt.close()


   KEY = [m for m in ["tab_clin_radiomics_lr", "ct_only", "pet_only", "early", "intermediate", "cross_attn_md", "ens_soft_fusion", "ens_hard_fusion"] if m in oof]
   fig, axes = plt.subplots(2, 4, figsize=(17, 8)); axes = axes.ravel()
   for ax, m in zip(axes, KEY):
       cm = confusion_matrix(oof[m]["label"], (oof[m]["prob"] >= 0.5).astype(int), labels=[0, 1])
       sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax, xticklabels=["pred 0", "pred 1"], yticklabels=["true 0", "true 1"]); ax.set_title(m, fontsize=10)
   for ax in axes[len(KEY):]:
       ax.axis("off")
   plt.tight_layout(); plt.savefig(f"{FIG_DIR}/eval4_confusion_matrices.png", dpi=140); plt.close()


   plt.figure(figsize=(6.5, 6))
   for i, m in enumerate(KEY):
       try:
           fp, mp = calibration_curve(oof[m]["label"], oof[m]["prob"], n_bins=5, strategy="quantile")
           plt.plot(mp, fp, marker="o", color=pal[i], label=f"{m} (Brier {RES.loc[m, 'brier']:.2f})")
       except ValueError:
           pass
   plt.plot([0, 1], [0, 1], "k:"); plt.xlabel("mean predicted probability"); plt.ylabel("observed fraction positive")
   plt.title("Calibration (quantile bins; few patients -> noisy)"); plt.legend(fontsize=7); plt.tight_layout()
   plt.savefig(f"{FIG_DIR}/eval5_calibration.png", dpi=140); plt.close()


   hist_models = [m for m in ORDER if m in hist]
   if hist_models:
       nc = 4; nr = int(np.ceil(len(hist_models) / nc))
       fig, axes = plt.subplots(nr, nc, figsize=(4.6 * nc, 3.6 * nr), squeeze=False); axes = axes.ravel()
       for ax, m in zip(axes, hist_models):
           h = hist[m]
           for _, g in h.groupby("fold"):
               ax.plot(g["epoch"], g["train_loss"], color="#4C72B0", alpha=0.25); ax.plot(g["epoch"], g["val_loss"], color="#DD8452", alpha=0.25)
           mm = h.groupby("epoch")[["train_loss", "val_loss"]].mean()
           ax.plot(mm.index, mm["train_loss"], color="#4C72B0", lw=2.5, label="train"); ax.plot(mm.index, mm["val_loss"], color="#DD8452", lw=2.5, label="val")
           ax.set_title(m, fontsize=10); ax.set_xlabel("epoch"); ax.legend(fontsize=8)
       for ax in axes[len(hist_models):]:
           ax.axis("off")
       plt.suptitle("Training vs validation loss (thin = folds, thick = mean)"); plt.tight_layout()
       plt.savefig(f"{FIG_DIR}/eval6_training_curves.png", dpi=140); plt.close()


   ABL = {"Modality (single vs both)": ["tab_clin_radiomics_lr", "ct_only", "pet_only", "early"],
          "Fusion strategy": ["early", "intermediate", "cross_attn", "late_ct_pet"],
          "Modality dropout (proposed)": ["cross_attn", "cross_attn_md"],
          "Adding clinical features": ["intermediate", "intermediate_clin", "cross_attn_md", "cross_attn_md_clin"],
          "Ensembles": ["ens_soft_fusion", "ens_hard_fusion", "ens_soft_image_all", "ens_hard_image_all"]}
   print("\n=== Ablation view (ROC-AUC [95% CI] | PR-AUC) ===")
   for title, ms in ABL.items():
       print(f"\n{title}")
       for m in [x for x in ms if x in RES.index]:
           r = RES.loc[m]
           print(f"  {m:24s} {r['roc_auc']:.3f} [{r['roc_auc_lo']:.3f}, {r['roc_auc_hi']:.3f}] | PR-AUC {r['pr_auc']:.3f}")




if __name__ == "__main__":
   main()
