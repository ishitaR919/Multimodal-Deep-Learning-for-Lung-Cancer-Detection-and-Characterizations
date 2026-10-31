"""Step 8 - missing-modality robustness: the same trained models, with one modality blanked at test time.


Usage:   python 08_missing_modality.py
Inputs:  RES_DIR/oof_<model>.csv, RES_DIR/oof_miss_<model>_<mode>.csv (step 5)
Outputs: RES_DIR/results_missing_modality.csv, FIG_DIR/eval6_missing_modality.png
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402


from config import RES_DIR, FIG_DIR, MODEL_NAMES  # noqa: E402
from common import load_oof, load_oof_miss, bootstrap_ci, roc_auc_safe  # noqa: E402




def main():
   OOF, OOF_MISS = load_oof(MODEL_NAMES), load_oof_miss(MODEL_NAMES)
   OOF_MISS = {m: d for m, d in OOF_MISS.items() if m in OOF}
   if not OOF_MISS:
       raise SystemExit("No missing-modality predictions found - run 05_train.py first.")
   rows = []
   for m, d in OOF_MISS.items():
       for avail, df_ in [("all", OOF[m])] + [(f"{k} missing", v) for k, v in d.items()]:
           lo, hi = bootstrap_ci(df_["label"], df_["prob"], roc_auc_score, 500)
           rows.append({"model": m, "available": avail, "roc_auc": roc_auc_safe(df_["label"], df_["prob"]), "lo": lo, "hi": hi})
   MISS = pd.DataFrame(rows)
   MISS.to_csv(f"{RES_DIR}/results_missing_modality.csv", index=False)
   print(MISS.pivot(index="model", columns="available", values="roc_auc").round(3).to_string())
   conds = ["all", "ct missing", "rad missing", "clin missing"]
   models_m = list(OOF_MISS.keys())
   fig, ax = plt.subplots(figsize=(11, 5)); w = 0.2
   for k, cond in enumerate(conds):
       sub = MISS[MISS["available"] == cond].set_index("model").reindex(models_m)
       ax.bar(np.arange(len(models_m)) + (k - 1.5) * w, sub["roc_auc"], w,
              yerr=[sub["roc_auc"] - sub["lo"], sub["hi"] - sub["roc_auc"]], capsize=2, label=cond)
   ax.axhline(0.5, color="k", ls=":"); ax.set_xticks(range(len(models_m))); ax.set_xticklabels(models_m, rotation=15)
   ax.set_ylabel("ROC-AUC"); ax.legend(); ax.set_title("Robustness when a modality is missing at test time")
   plt.tight_layout(); plt.savefig(f"{FIG_DIR}/eval6_missing_modality.png", dpi=140); plt.close()




if __name__ == "__main__":
   main()