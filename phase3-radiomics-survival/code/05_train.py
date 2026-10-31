"""Step 5 - train the image / fusion models with patient-level cross-validation.


Models: ct_cnn (CT only), ct_rad_mid and ct_rad_clin_mid (intermediate fusion), gated (attention-style gate),
gated_md (proposed: gate + modality dropout). Each fold also predicts with one modality blanked out.


Usage:   python 05_train.py                     # all models
        python 05_train.py ct_cnn gated_md     # selected models
Inputs:  RES_DIR/cohort.csv, CROP_DIR/*.npz
Outputs: RES_DIR/oof_<model>.csv, RES_DIR/history_<model>.csv, RES_DIR/oof_miss_<model>_<ct|rad|clin>.csv,
        WEIGHT_DIR/<model>_fold0.pt (models in SAVE_WEIGHTS)
"""
import sys
import time


import pandas as pd


from config import RES_DIR, N_FOLDS, MODEL_NAMES
from common import load_state, torch_parts, roc_auc_safe, compute_metrics, c_index




def run_cv(T, st, name):
   t_start = time.time()
   tests, hists, miss_all = [], [], {}
   for fold in range(N_FOLDS):
       t0 = time.time()
       te_df, h, miss, sizes = T["train_fold"](name, st, fold)
       tests.append(te_df); hists.append(h)
       for k, v in miss.items():
           miss_all.setdefault(k, []).append(v)
       print(f"[{name}] fold {fold + 1}/{N_FOLDS}: train/val/test {sizes} | test AUC "
             f"{roc_auc_safe(te_df['label'], te_df['prob']):.3f} | best val loss {h['val_loss'].min():.3f} | {time.time() - t0:.0f}s")
   oof = pd.concat(tests, ignore_index=True)
   oof.to_csv(f"{RES_DIR}/oof_{name}.csv", index=False)
   pd.concat(hists, ignore_index=True).to_csv(f"{RES_DIR}/history_{name}.csv", index=False)
   for mode, dfs in miss_all.items():
       pd.concat(dfs, ignore_index=True).to_csv(f"{RES_DIR}/oof_miss_{name}_{mode}.csv", index=False)
   m = compute_metrics(oof["label"], oof["prob"])
   ci = c_index(oof["patient_id"].map(st.time), oof["patient_id"].map(st.event), oof["prob"])
   print(f"==> {name}: pooled OOF ROC-AUC {m['roc_auc']:.3f} | PR-AUC {m['pr_auc']:.3f} | C-index {ci:.3f} | "
         f"sens {m['sensitivity']:.3f} | spec {m['specificity']:.3f}   ({(time.time() - t_start) / 60:.1f} min)")




def main():
   names = sys.argv[1:] or MODEL_NAMES
   unknown = [n for n in names if n not in MODEL_NAMES]
   if unknown:
       raise SystemExit(f"Unknown model(s) {unknown}; choose from {MODEL_NAMES}")
   st = load_state()
   T = torch_parts()
   print("Device:", T["device"])
   for name in names:
       run_cv(T, st, name)




if __name__ == "__main__":
   main()