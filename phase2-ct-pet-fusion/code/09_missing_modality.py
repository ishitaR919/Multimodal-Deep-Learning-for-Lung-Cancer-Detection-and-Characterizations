"""Step 9 - missing-modality robustness: the same trained fusion models, with CT or PET blanked at test time.
Models trained with modality dropout should degrade less.


Usage:  python 09_missing_modality.py


Input:  oof_<model>.csv and oof_miss_<model>_<ct|pet>.csv (step 6)
Output (RES_DIR): results_missing_modality.csv
Output (FIG_DIR): eval7_missing_modality.png
"""
import os


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score


from config import RES_DIR, FIG_DIR, MODEL_SPECS
from common import load_oof, bootstrap_ci, roc_auc_safe, compute_metrics




def main():
   both = [m for m, s in MODEL_SPECS.items() if s["both"]]
   oof = load_oof(both)
   oof_miss = {m: {k: pd.read_csv(f"{RES_DIR}/oof_miss_{m}_{k}.csv") for k in ("ct", "pet")}
               for m in both if m in oof and all(os.path.exists(f"{RES_DIR}/oof_miss_{m}_{k}.csv") for k in ("ct", "pet"))}
   if oof_miss:
       rows = []
       for m, d in oof_miss.items():
           full = oof[m]
           for label_, df_ in [("both", full), ("PET only (CT missing)", d["ct"]), ("CT only (PET missing)", d["pet"])]:
               lo, hi = bootstrap_ci(df_["label"], df_["prob"], roc_auc_score, 500)
               rows.append({"model": m, "available": label_, "roc_auc": roc_auc_safe(df_["label"], df_["prob"]), "lo": lo, "hi": hi,
                            "balanced_acc": compute_metrics(df_["label"], df_["prob"])["balanced_acc"]})
       MISS = pd.DataFrame(rows)
       MISS.to_csv(f"{RES_DIR}/results_missing_modality.csv", index=False)
       print(MISS.pivot(index="model", columns="available", values="roc_auc").round(3).to_string())
       models_m = list(oof_miss.keys())
       fig, ax = plt.subplots(figsize=(11, 5)); w = 0.26
       for k, (lab, color) in enumerate([("both", "#55A868"), ("PET only (CT missing)", "#DD8452"), ("CT only (PET missing)", "#4C72B0")]):
           sub = MISS[MISS["available"] == lab].set_index("model").loc[models_m]
           ax.bar(np.arange(len(models_m)) + (k - 1) * w, sub["roc_auc"], w, yerr=[sub["roc_auc"] - sub["lo"], sub["hi"] - sub["roc_auc"]], capsize=2, label=lab, color=color)
       ax.axhline(0.5, color="k", ls=":"); ax.set_xticks(range(len(models_m))); ax.set_xticklabels(models_m, rotation=20); ax.set_ylabel("ROC-AUC"); ax.legend()
       ax.set_title("Robustness when a modality is missing at test time (models with modality dropout should degrade less)")
       plt.tight_layout(); plt.savefig(f"{FIG_DIR}/eval7_missing_modality.png", dpi=140); plt.close()
   else:
       print("No fusion model has been trained yet - run 06_train.py first.")




if __name__ == "__main__":
   main()
