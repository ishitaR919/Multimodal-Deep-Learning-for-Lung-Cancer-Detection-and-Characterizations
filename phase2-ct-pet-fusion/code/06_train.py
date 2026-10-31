"""Step 6 - train the image models with patient-level cross-validation.


Models (config.MODEL_SPECS): ct_only, pet_only, early, intermediate, intermediate_clin, cross_attn,
cross_attn_md (proposed: cross-modal attention + modality dropout), cross_attn_md_clin.


Usage:  python 06_train.py                          # all models
       python 06_train.py ct_only cross_attn_md    # selected models


Input:  crops (step 3), clinical columns
Output (RES_DIR):    oof_<model>.csv, history_<model>.csv,
                    oof_miss_<model>_<ct|pet>.csv (fusion models: one modality blanked at test time)
Output (WEIGHT_DIR): <model>_fold0.pt for config.SAVE_WEIGHTS (used by 10_explain.py)
"""
import sys
import time


import pandas as pd
from sklearn.model_selection import train_test_split


from config import SEED, N_FOLDS, RES_DIR, MODEL_SPECS
from common import load_state, make_clin_dict, compute_metrics, roc_auc_safe
from models import DEVICE, train_fold




def run_cv(st, name):
   spec, t_start = MODEL_SPECS[name], time.time()
   tests, hists, miss_all = [], [], {"ct": [], "pet": []}
   for fold in range(N_FOLDS):
       te = [p for p in st.pids if st.fold_of[p] == fold]
       trv = [p for p in st.pids if st.fold_of[p] != fold]
       try:
           tr, va = train_test_split(trv, test_size=0.15, stratify=[st.labels[p] for p in trv], random_state=SEED + fold)
       except ValueError:                                  # tiny quick-test cohorts
           tr, va = train_test_split(trv, test_size=0.15, random_state=SEED + fold)
       clin_dict, n_clin = make_clin_dict(st.clin_raw, tr, st.pids) if spec["clin"] else (None, 0)
       t0 = time.time()
       te_df, h, miss = train_fold(name, fold, tr, va, te, st.crops, st.labels, clin_dict, n_clin)
       tests.append(te_df); hists.append(h)
       for k, v in miss.items():
           miss_all[k].append(v)
       print(f"[{name}] fold {fold + 1}/{N_FOLDS}: train {len(tr)} val {len(va)} test {len(te)} | "
             f"test AUC {roc_auc_safe(te_df['label'], te_df['prob']):.3f} | best val loss {h['val_loss'].min():.3f} | {time.time() - t0:.0f}s")
   oof, hist = pd.concat(tests, ignore_index=True), pd.concat(hists, ignore_index=True)
   oof.to_csv(f"{RES_DIR}/oof_{name}.csv", index=False)
   hist.to_csv(f"{RES_DIR}/history_{name}.csv", index=False)
   if spec["both"]:
       for k, v in miss_all.items():
           pd.concat(v, ignore_index=True).to_csv(f"{RES_DIR}/oof_miss_{name}_{k}.csv", index=False)
   m = compute_metrics(oof["label"], oof["prob"])
   print(f"==> {name}: pooled OOF ROC-AUC {m['roc_auc']:.3f} | PR-AUC {m['pr_auc']:.3f} | balanced acc {m['balanced_acc']:.3f} "
         f"| sens {m['sensitivity']:.3f} | spec {m['specificity']:.3f}   ({(time.time() - t_start) / 60:.1f} min)")




def main():
   names = sys.argv[1:] or list(MODEL_SPECS)
   unknown = [n for n in names if n not in MODEL_SPECS]
   if unknown:
       sys.exit(f"Unknown model(s): {unknown}. Choose from {list(MODEL_SPECS)}")
   print("Device:", DEVICE, "(training on CPU works but is slow)" if DEVICE.type == "cpu" else "")
   st = load_state()
   for name in names:
       run_cv(st, name)




if __name__ == "__main__":
   main()
