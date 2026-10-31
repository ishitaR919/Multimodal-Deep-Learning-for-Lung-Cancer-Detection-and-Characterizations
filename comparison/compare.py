"""Phase 1 vs Phase 2 comparison, run locally on the downloaded Kaggle bundles.


Usage (after unzipping both bundles):
   python compare_phases.py --p1 path/to/phase1_bundle --p2 path/to/phase2_bundle --out comparison


Expected layout:
   <p1>/model_results/final_metrics_all_models_and_ensembles.csv
   <p2>/phase2_results/results_all_models.csv and oof_<model>.csv


Outputs (in --out):
   phase_summary.csv / .tex      side-by-side headline numbers (DIFFERENT tasks: context only, not a head-to-head test)
   fusion_vs_ct.csv / .tex       Phase 2: AUC gain of every model over CT-only, paired patient bootstrap 95% CI + p
   fusion_vs_ct.png              forest plot of those gains
"""
import argparse
import os


import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score


BASELINE = "ct_only"
B = 2000
SEED = 0


LABELS = {"ct_only": "CT only", "pet_only": "PET only", "early": "Early fusion", "intermediate": "Intermediate fusion",
         "intermediate_clin": "Intermediate + clinical", "cross_attn": "Cross-attention",
         "cross_attn_md": "Cross-attention + modality dropout (proposed)",
         "cross_attn_md_clin": "Proposed + clinical", "late_ct_pet": "Late fusion (CT, PET)",
         "late_ct_pet_clin": "Late fusion (CT, PET, clinical)", "ens_soft_fusion": "Soft-vote ensemble (fusion)",
         "ens_hard_fusion": "Hard-vote ensemble (fusion)", "ens_soft_image_all": "Soft-vote ensemble (all image)",
         "ens_hard_image_all": "Hard-vote ensemble (all image)", "tab_clinical_lr": "Clinical (LR)",
         "tab_radiomics_lr": "Tumour ROI features (LR)", "tab_clin_radiomics_lr": "Clinical + ROI features (LR)",
         "tab_clin_radiomics_rf": "Clinical + ROI features (RF)"}




def paired_delta(base, other, metric):
   """Paired bootstrap over patients present in both models: delta = metric(other) - metric(base)."""
   m = base.merge(other, on="patient_id", suffixes=("_b", "_o"))
   y, pb, po = m["label_b"].values, m["prob_b"].values, m["prob_o"].values
   point = metric(y, po) - metric(y, pb)
   rng, d = np.random.RandomState(SEED), []
   for _ in range(B):
       i = rng.randint(0, len(y), len(y))
       if len(np.unique(y[i])) > 1:
           d.append(metric(y[i], po[i]) - metric(y[i], pb[i]))
   d = np.asarray(d)
   lo, hi = np.percentile(d, [2.5, 97.5])
   p = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))       # two-sided bootstrap p for delta = 0
   return point, lo, hi, p, len(y)




def to_latex(df, path, caption, label, fmt=3):
   cols = "l" + "c" * (df.shape[1])
   lines = [r"\begin{table}[H]", r"\centering", r"\small", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
            rf"\begin{{tabular}}{{{cols}}}", r"\toprule",
            " & ".join([""] + [c.replace("_", r"\_") for c in df.columns]) + r" \\", r"\midrule"]
   for idx, row in df.iterrows():
       cells = [f"{v:.{fmt}f}" if isinstance(v, (float, np.floating)) else str(v) for v in row.values]
       lines.append(" & ".join([str(idx).replace("_", r"\_")] + cells) + r" \\")
   lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
   open(path, "w").write("\n".join(lines) + "\n")




def main():
   ap = argparse.ArgumentParser()
   ap.add_argument("--p1", required=True, help="unzipped phase1_results_bundle folder")
   ap.add_argument("--p2", required=True, help="unzipped phase2_results_bundle folder")
   ap.add_argument("--out", default="comparison")
   a = ap.parse_args()
   os.makedirs(a.out, exist_ok=True)


   p1 = pd.read_csv(os.path.join(a.p1, "model_results", "final_metrics_all_models_and_ensembles.csv"), index_col=0)
   p2_dir = os.path.join(a.p2, "phase2_results")
   p2 = pd.read_csv(os.path.join(p2_dir, "results_all_models.csv"), index_col=0)


   # ---- 1. side-by-side headline numbers (context only) ----
   p1_best_single = p1.drop(index=[i for i in p1.index if "vote" in i])["roc_auc"].astype(float).idxmax()
   p2_img = [m for m in p2.index if not m.startswith("tab_")]
   p2_best = p2.loc[p2_img, "roc_auc"].astype(float).idxmax()
   rows = []
   for phase, task, unit, proto, df, models in [
       ("Phase 1", "nodule vs non-nodule (LUNA16, CT)", "candidate", "single 70/15/15 split", p1,
        [p1_best_single, "soft_vote"]),
       ("Phase 2", "adenocarcinoma vs other (LUNG-PET-CT-Dx)", "patient", "5-fold CV (pooled out-of-fold)", p2,
        [BASELINE, p2_best])]:
       for m in models:
           if m not in df.index:
               continue
           r = df.loc[m]
           rows.append({"phase": phase, "task": task, "unit": unit, "protocol": proto, "model": LABELS.get(m, m),
                        "roc_auc": float(r["roc_auc"]), "pr_auc": float(r["pr_auc"]),
                        "sensitivity": float(r.get("recall_sensitivity", r.get("sensitivity", np.nan))),
                        "specificity": float(r["specificity"]), "brier": float(r.get("brier", np.nan))})
   summary = pd.DataFrame(rows)
   summary.to_csv(os.path.join(a.out, "phase_summary.csv"), index=False)
   to_latex(summary.set_index("model")[["phase", "roc_auc", "pr_auc", "sensitivity", "specificity", "brier"]],
            os.path.join(a.out, "phase_summary.tex"),
            "Headline results per phase. The phases use different datasets, labels and units, so the rows "
            "describe each phase and are not a head-to-head comparison.", "tab:phase_summary")
   print("=== Per-phase headline results (different tasks - context only) ===")
   print(summary[["phase", "model", "roc_auc", "pr_auc", "sensitivity", "specificity"]].round(3).to_string(index=False))


   # ---- 2. Phase 2: does adding PET / clinical data beat CT alone? (same patients, same folds) ----
   base_path = os.path.join(p2_dir, f"oof_{BASELINE}.csv")
   if not os.path.exists(base_path):
       raise FileNotFoundError(f"{base_path} missing - Phase 2 CT-only model was not trained.")
   base = pd.read_csv(base_path)[["patient_id", "label", "prob"]]
   rows = []
   for m in p2.index:
       f = os.path.join(p2_dir, f"oof_{m}.csv")
       if m == BASELINE or not os.path.exists(f):
           continue
       other = pd.read_csv(f)[["patient_id", "label", "prob"]]
       d_auc, lo, hi, p, n = paired_delta(base, other, roc_auc_score)
       d_ap, ap_lo, ap_hi, ap_p, _ = paired_delta(base, other, average_precision_score)
       rows.append({"model": LABELS.get(m, m), "roc_auc": float(p2.loc[m, "roc_auc"]), "delta_roc_auc": d_auc,
                    "ci_lo": lo, "ci_hi": hi, "p_value": p, "delta_pr_auc": d_ap, "pr_ci_lo": ap_lo,
                    "pr_ci_hi": ap_hi, "n_patients": n})
   gains = pd.DataFrame(rows).sort_values("delta_roc_auc", ascending=False).set_index("model")
   gains.to_csv(os.path.join(a.out, "fusion_vs_ct.csv"))
   tab = gains[["roc_auc", "delta_roc_auc", "ci_lo", "ci_hi", "p_value", "delta_pr_auc"]].copy()
   tab.columns = ["ROC-AUC", "dAUC vs CT", "95% CI low", "95% CI high", "p", "dPR-AUC vs CT"]
   to_latex(tab, os.path.join(a.out, "fusion_vs_ct.tex"),
            f"Phase 2: change in ROC-AUC and PR-AUC compared with the CT-only model (ROC-AUC "
            f"{float(p2.loc[BASELINE, 'roc_auc']):.3f}), on the same patients and folds. 95\\% CI and p-value "
            f"from a paired bootstrap over patients ({B} resamples).", "tab:fusion_vs_ct")
   print(f"\n=== Phase 2: gain over CT-only (paired bootstrap, {B} resamples) ===")
   print(gains[["roc_auc", "delta_roc_auc", "ci_lo", "ci_hi", "p_value", "delta_pr_auc"]].round(3).to_string())


   fig, ax = plt.subplots(figsize=(8, 0.45 * len(gains) + 1.5))
   y = np.arange(len(gains))
   sig = gains["ci_lo"] > 0
   ax.errorbar(gains["delta_roc_auc"], y, xerr=[gains["delta_roc_auc"] - gains["ci_lo"], gains["ci_hi"] - gains["delta_roc_auc"]],
               fmt="none", ecolor="#888888", capsize=3)
   ax.scatter(gains["delta_roc_auc"], y, c=np.where(sig, "#2a7a3b", "#4C72B0"), zorder=3)
   ax.axvline(0, color="k", ls=":")
   ax.set_yticks(y); ax.set_yticklabels(gains.index); ax.invert_yaxis()
   ax.set_xlabel("ROC-AUC difference vs CT-only (95% CI)")
   ax.set_title("Phase 2: what each model adds over CT alone\n(green = whole CI above 0)")
   plt.tight_layout(); plt.savefig(os.path.join(a.out, "fusion_vs_ct.png"), dpi=150)
   print(f"\nSaved tables and figure to {os.path.abspath(a.out)}")




if __name__ == "__main__":
   main()