"""Step 5 - patient-level CV folds and tabular baselines (clinical / tumour-ROI "radiomics-lite" / both).


Usage:  python 05_tabular_baselines.py


Input:  crops + features (step 3), clinical columns in paired_patients_v2.csv
Output (RES_DIR): oof_tab_clinical_lr.csv, oof_tab_radiomics_lr.csv, oof_tab_clin_radiomics_lr.csv,
                 oof_tab_clin_radiomics_rf.csv  (out-of-fold patient predictions)
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


from config import SEED, N_FOLDS, RES_DIR, CLIN_COLS, RAD_COLS
from common import load_state, summarize_oof




def run_tabular_cv(st, name, cols, kind="lr"):
   X = st.master.set_index("patient_id").loc[st.pids, cols]
   y = np.array([st.labels[p] for p in st.pids])
   fold_arr = np.array([st.fold_of[p] for p in st.pids])
   rows = []
   for f in range(N_FOLDS):
       te = fold_arr == f
       if kind == "lr":
           est = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(),
                               LogisticRegression(C=0.5, class_weight="balanced", max_iter=2000))
       else:
           est = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                               RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                                      class_weight="balanced_subsample", random_state=SEED))
       est.fit(X[~te], y[~te])
       rows.append(pd.DataFrame({"patient_id": np.array(st.pids)[te], "fold": f, "label": y[te],
                                 "prob": est.predict_proba(X[te])[:, 1]}))
   oof = pd.concat(rows, ignore_index=True)
   oof.to_csv(f"{RES_DIR}/oof_{name}.csv", index=False)
   return oof




def main():
   st = load_state(with_crops=False)
   print(f"MASTER: {len(st.master)} patients | label counts {st.master['label'].value_counts().sort_index().to_dict()}")
   print("Patients per fold / positives per fold:")
   print(pd.DataFrame({"fold": [st.fold_of[p] for p in st.pids], "label": [st.labels[p] for p in st.pids]})
         .groupby("fold")["label"].agg(patients="count", positives="sum").to_string())
   print("\nClinical NaN count (imputed with train-fold medians):", st.clin_raw.isna().sum().to_dict())


   oof = {"tab_clinical_lr": run_tabular_cv(st, "tab_clinical_lr", CLIN_COLS, "lr"),
          "tab_radiomics_lr": run_tabular_cv(st, "tab_radiomics_lr", RAD_COLS, "lr"),
          "tab_clin_radiomics_lr": run_tabular_cv(st, "tab_clin_radiomics_lr", CLIN_COLS + RAD_COLS, "lr"),
          "tab_clin_radiomics_rf": run_tabular_cv(st, "tab_clin_radiomics_rf", CLIN_COLS + RAD_COLS, "rf")}
   tab = summarize_oof(oof, B=300)
   print("Tabular baselines (pooled out-of-fold, patient level):")
   print(tab[["roc_auc", "pr_auc", "balanced_acc", "sensitivity", "specificity"]].round(3).to_string())




if __name__ == "__main__":
   main()