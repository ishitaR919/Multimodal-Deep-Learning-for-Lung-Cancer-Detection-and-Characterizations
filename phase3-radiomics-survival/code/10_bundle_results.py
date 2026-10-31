"""Step 10 - run summary + zip of results and figures (crops and volumes excluded: regenerable).


Usage:   python 10_bundle_results.py
Inputs:  RES_DIR, FIG_DIR
Outputs: RES_DIR/run_summary.json, WORK_DIR/phase3_results_bundle.zip
"""
import json
import os
import shutil


from config import (WORK_DIR, RES_DIR, FIG_DIR, QUICK_TEST, FEATURE_SET, N_FOLDS, EPOCHS, HORIZON_DAYS, K_RAD, SEED,
                   ORDER)
from common import load_state, load_oof




def main():
   st = load_state(verbose=False)
   summary = {"quick_test": QUICK_TEST, "feature_set": FEATURE_SET, "n_patients": len(st.pids), "n_folds": N_FOLDS,
              "epochs": EPOCHS, "horizon_days": HORIZON_DAYS, "k_rad": K_RAD, "n_radiomic_features": len(st.rad_cols),
              "label_counts": {str(k): int(v) for k, v in st.master["label"].value_counts().items()},
              "models_trained": list(load_oof(ORDER)), "seed": SEED}
   json.dump(summary, open(f"{RES_DIR}/run_summary.json", "w"), indent=2)


   stage = f"{WORK_DIR}/_zip_stage3"
   shutil.rmtree(stage, ignore_errors=True); os.makedirs(stage)
   shutil.copytree(RES_DIR, f"{stage}/phase3_results", ignore=shutil.ignore_patterns("crops_*", "*.npz"))
   shutil.copytree(FIG_DIR, f"{stage}/figures_phase3")
   zp = shutil.make_archive(f"{WORK_DIR}/phase3_results_bundle", "zip", stage)
   shutil.rmtree(stage)
   print(f"Created {zp} ({os.path.getsize(zp) / 1e6:.0f} MB)")
   if QUICK_TEST:
       print("QUICK_TEST run: these numbers only prove the pipeline works. Re-run with QUICK_TEST=0.")




if __name__ == "__main__":
   main()
