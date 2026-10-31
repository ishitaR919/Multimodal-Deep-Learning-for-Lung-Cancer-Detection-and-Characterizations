"""Step 4 - tabular baselines on the same patient-level folds: clinical, tumour volume, radiomics, clinical + radiomics.


Feature selection (variance, correlation filter, top-K univariate) is fitted on the training folds only.


Usage:   python 04_tabular_baselines.py
Inputs:  RES_DIR/cohort.csv, CROP_DIR/*.npz (radiomic features)
Outputs: RES_DIR/oof_tab_*.csv (pooled out-of-fold patient predictions)
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


from config import RES_DIR, N_FOLDS, SEED
from common import load_state, tab_estimator, summarize_oof




def run_tabular_cv(st, name, cols, kind="lr"):
   y = np.array([st.labels[p] for p in st.pids])
   fold_arr = np.array([st.fold_of[p] for p in st.pids])
   X = st.xtab.loc[st.pids, cols]
   rows = []
   for f in range(N_FOLDS):
       te = fold_arr == f
       clf = (LogisticRegression(C=0.5, class_weight="balanced", max_iter=5000) if kind == "lr" else
              RandomForestClassifier(n_estimators=300, min_samples_leaf=3, class_weight="balanced_subsample", random_state=SEED))
       est = tab_estimator(cols, clf).fit(X[~te], y[~te])
       rows.append(pd.DataFrame({"patient_id": np.array(st.pids)[te], "fold": f, "label": y[te],
                                 "prob": est.predict_proba(X[te])[:, 1]}))
   oof = pd.concat(rows, ignore_index=True)
   oof.to_csv(f"{RES_DIR}/oof_{name}.csv", index=False)
   return oof




def main():
   st = load_state()
   print(pd.DataFrame({"fold": [st.fold_of[p] for p in st.pids], "label": [st.labels[p] for p in st.pids]})
         .groupby("fold")["label"].agg(patients="count", positives="sum").to_string())
   oof = {
       "tab_clinical_lr": run_tabular_cv(st, "tab_clinical_lr", st.clin_cols),
       "tab_volume_lr": run_tabular_cv(st, "tab_volume_lr", ["log_gtv_volume"]),        # is radiomics more than size?
       "tab_radiomics_lr": run_tabular_cv(st, "tab_radiomics_lr", st.rad_cols),
       "tab_clin_radiomics_lr": run_tabular_cv(st, "tab_clin_radiomics_lr", st.clin_cols + st.rad_cols),  # early fusion
       "tab_clin_radiomics_rf": run_tabular_cv(st, "tab_clin_radiomics_rf", st.clin_cols + st.rad_cols, "rf"),
   }
   res = summarize_oof(st, oof, B=300)
   print("Tabular baselines (pooled out-of-fold, patient level):")
   print(res[["roc_auc", "pr_auc", "c_index", "balanced_acc", "sensitivity", "specificity"]].astype(float).round(3).to_string())




if __name__ == "__main__":
   main()