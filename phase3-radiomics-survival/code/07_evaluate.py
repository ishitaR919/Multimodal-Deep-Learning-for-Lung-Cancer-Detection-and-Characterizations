"""Step 7 - full evaluation: results table with bootstrap CIs and C-index, ROC / PR curves, CI bars,
Kaplan-Meier risk groups, calibration, training curves and the ablation view.


Usage:   python 07_evaluate.py
Inputs:  RES_DIR/oof_*.csv, RES_DIR/history_*.csv
Outputs: RES_DIR/results_all_models.csv, FIG_DIR/eval1_roc_pr.png ... eval5_training_curves.png
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.metrics import roc_curve, precision_recall_curve  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402


from config import RES_DIR, FIG_DIR, ORDER as ALL_ORDER  # noqa: E402
from common import load_state, load_oof, load_hist, summarize_oof, km_curve, logrank_p  # noqa: E402


sns.set_style("whitegrid")


ABL = {"Single modality": ["tab_clinical_lr", "tab_volume_lr", "tab_radiomics_lr", "ct_cnn"],
      "Adding radiomics to CT": ["ct_cnn", "ct_rad_mid", "late_ct_rad"],
      "Adding clinical data": ["tab_radiomics_lr", "tab_clin_radiomics_lr", "ct_rad_mid", "ct_rad_clin_mid", "late_ct_rad_clin"],
      "Fusion strategy (CT + rad + clin)": ["tab_clin_radiomics_lr", "ct_rad_clin_mid", "gated", "late_ct_rad_clin"],
      "Modality dropout (proposed)": ["gated", "gated_md"]}




def main():
   st = load_state()
   OOF, HIST = load_oof(ALL_ORDER), load_hist(ALL_ORDER)
   ORDER = [m for m in ALL_ORDER if m in OOF]
   if not ORDER:
       raise SystemExit("No out-of-fold predictions found - run steps 04-06 first.")
   RES = summarize_oof(st, {m: OOF[m] for m in ORDER}, B=1000)
   RES.to_csv(f"{RES_DIR}/results_all_models.csv")
   print("=== Pooled out-of-fold results (patient level; threshold 0.5; 95% CI by patient bootstrap) ===")
   print(RES[["roc_auc", "roc_auc_lo", "roc_auc_hi", "pr_auc", "c_index", "balanced_acc", "sensitivity", "specificity",
              "brier"]].astype(float).round(3).to_string())
   prev = float(np.mean([st.labels[p] for p in st.pids]))
   print(f"\nPositive-class prevalence = {prev:.3f}  (PR-AUC of a random classifier ~ {prev:.3f})")


   pal = sns.color_palette("tab10", 12)
   KEY = [m for m in ["tab_clinical_lr", "tab_volume_lr", "tab_radiomics_lr", "ct_cnn", "ct_rad_clin_mid", "gated_md",
                      "late_ct_rad_clin"] if m in OOF]
   fig, axes = plt.subplots(1, 2, figsize=(14, 6))
   for i, m in enumerate(KEY):
       d = OOF[m]
       fpr, tpr, _ = roc_curve(d["label"], d["prob"]); axes[0].plot(fpr, tpr, lw=2, color=pal[i], label=f"{m} ({RES.loc[m, 'roc_auc']:.2f})")
       pr, rc, _ = precision_recall_curve(d["label"], d["prob"]); axes[1].plot(rc, pr, lw=2, color=pal[i], label=f"{m} ({RES.loc[m, 'pr_auc']:.2f})")
   axes[0].plot([0, 1], [0, 1], "k:"); axes[0].set_xlabel("1 - specificity"); axes[0].set_ylabel("sensitivity"); axes[0].set_title("ROC (pooled out-of-fold)")
   axes[1].axhline(prev, color="k", ls=":"); axes[1].set_xlabel("recall"); axes[1].set_ylabel("precision"); axes[1].set_title("Precision-Recall (pooled out-of-fold)")
   for a in axes:
       a.legend(fontsize=8)
   plt.tight_layout(); plt.savefig(f"{FIG_DIR}/eval1_roc_pr.png", dpi=140); plt.close()


   fig, axes = plt.subplots(1, 2, figsize=(15, max(5, 0.4 * len(ORDER))))
   for ax, (metric, lo, hi, base) in zip(axes, [("roc_auc", "roc_auc_lo", "roc_auc_hi", 0.5), ("pr_auc", "pr_auc_lo", "pr_auc_hi", prev)]):
       r = RES.loc[ORDER, [metric, lo, hi]].astype(float)
       y = np.arange(len(r))
       ax.barh(y, r[metric], xerr=[r[metric] - r[lo], r[hi] - r[metric]], capsize=3,
               color=["#8172B3" if m.startswith("late") else "#4C72B0" if m.startswith("tab") else "#55A868" for m in r.index])
       ax.set_yticks(y); ax.set_yticklabels(r.index); ax.invert_yaxis(); ax.axvline(base, color="k", ls=":"); ax.set_title(f"{metric} with 95% CI")
   plt.tight_layout(); plt.savefig(f"{FIG_DIR}/eval2_auc_bars_ci.png", dpi=140); plt.close()


   # Kaplan-Meier: high vs low predicted risk (split at the median out-of-fold probability), all follow-up
   KM_MODELS = [m for m in ["tab_clinical_lr", "tab_radiomics_lr", "ct_cnn", "gated_md"] if m in OOF]
   if KM_MODELS:
       fig, axes = plt.subplots(1, len(KM_MODELS), figsize=(4.6 * len(KM_MODELS), 4), squeeze=False); axes = axes[0]
       for ax, m in zip(axes, KM_MODELS):
           d = OOF[m]
           t, e = d["patient_id"].map(st.time).values, d["patient_id"].map(st.event).values
           g = (d["prob"] > d["prob"].median()).astype(int).values
           for grp, col, lab in ((0, "#4C72B0", "low risk"), (1, "#DD8452", "high risk")):
               x, s = km_curve(t[g == grp], e[g == grp]); ax.step(x / 365.25, s, where="post", color=col, label=f"{lab} (n={(g == grp).sum()})")
           ax.set_title(f"{m}\nlog-rank p={logrank_p(t, e, g):.2g}, C={RES.loc[m, 'c_index']:.2f}", fontsize=9)
           ax.set_xlabel("years"); ax.legend(fontsize=8)
       axes[0].set_ylabel("overall survival")
       plt.tight_layout(); plt.savefig(f"{FIG_DIR}/eval3_km_risk_groups.png", dpi=140); plt.close()


   plt.figure(figsize=(6.5, 6))
   for i, m in enumerate(KEY):
       try:
           fp, mp_ = calibration_curve(OOF[m]["label"], OOF[m]["prob"], n_bins=5, strategy="quantile")
           plt.plot(mp_, fp, marker="o", color=pal[i], label=f"{m} (Brier {RES.loc[m, 'brier']:.2f})")
       except ValueError:
           pass
   plt.plot([0, 1], [0, 1], "k:"); plt.xlabel("mean predicted probability"); plt.ylabel("observed fraction positive")
   plt.title("Calibration (quantile bins)"); plt.legend(fontsize=7); plt.tight_layout()
   plt.savefig(f"{FIG_DIR}/eval4_calibration.png", dpi=140); plt.close()


   hist_models = [m for m in ORDER if m in HIST]
   if hist_models:
       fig, axes = plt.subplots(1, len(hist_models), figsize=(4.4 * len(hist_models), 3.6), squeeze=False); axes = axes[0]
       for ax, m in zip(axes, hist_models):
           h = HIST[m]
           for _, g in h.groupby("fold"):
               ax.plot(g["epoch"], g["train_loss"], color="#4C72B0", alpha=0.25); ax.plot(g["epoch"], g["val_loss"], color="#DD8452", alpha=0.25)
           mm = h.groupby("epoch")[["train_loss", "val_loss"]].mean()
           ax.plot(mm.index, mm["train_loss"], color="#4C72B0", lw=2.5, label="train"); ax.plot(mm.index, mm["val_loss"], color="#DD8452", lw=2.5, label="val")
           ax.set_title(m, fontsize=10); ax.set_xlabel("epoch"); ax.legend(fontsize=8)
       plt.suptitle("Training vs validation loss (thin = folds, thick = mean)"); plt.tight_layout()
       plt.savefig(f"{FIG_DIR}/eval5_training_curves.png", dpi=140); plt.close()


   print("\n=== Ablation view (ROC-AUC [95% CI] | PR-AUC | C-index) ===")
   for title, ms in ABL.items():
       print(f"\n{title}")
       for m in [x for x in ms if x in RES.index]:
           r = RES.loc[m]
           print(f"  {m:24s} {r['roc_auc']:.3f} [{r['roc_auc_lo']:.3f}, {r['roc_auc_hi']:.3f}] | {r['pr_auc']:.3f} | {r['c_index']:.3f}")




if __name__ == "__main__":
   main()