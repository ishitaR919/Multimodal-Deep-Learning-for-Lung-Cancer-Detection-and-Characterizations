"""Step 11 - save a run summary and zip results + figures for download (volumes and crops excluded: several GB,
regenerable).


Usage:  python 11_bundle_results.py


Output: RES_DIR/run_summary.json, WORK_DIR/phase2_results_bundle.zip
"""
import json
import os
import shutil


from config import (TASK, QUICK_TEST, N_FOLDS, EPOCHS, CROP_FOV_MM, CROP_PX, SLICES_PER_PATIENT, SEED, MODEL_SPECS,
                   WORK_DIR, RES_DIR, FIG_DIR)
from common import load_state




def main():
   st = load_state(with_crops=False)
   summary = {"task": TASK, "quick_test": QUICK_TEST, "n_patients": len(st.pids), "n_folds": N_FOLDS, "epochs": EPOCHS,
              "label_counts": {str(k): int(v) for k, v in st.master["label"].value_counts().items()},
              "models_trained": [m for m in MODEL_SPECS if os.path.exists(f"{RES_DIR}/oof_{m}.csv")],
              "crop_fov_mm": CROP_FOV_MM, "crop_px": CROP_PX, "slices_per_patient": SLICES_PER_PATIENT, "seed": SEED}
   json.dump(summary, open(f"{RES_DIR}/run_summary.json", "w"), indent=2)


   stage = f"{WORK_DIR}/_zip_stage"
   shutil.rmtree(stage, ignore_errors=True); os.makedirs(stage)
   shutil.copytree(RES_DIR, f"{stage}/phase2_results", ignore=shutil.ignore_patterns("volumes", "crops", "*.nii.gz"))
   shutil.copytree(FIG_DIR, f"{stage}/figures_phase2")
   zp = shutil.make_archive(f"{WORK_DIR}/phase2_results_bundle", "zip", stage)
   shutil.rmtree(stage)
   print(f"Created {zp} ({os.path.getsize(zp) / 1e6:.0f} MB)")
   if QUICK_TEST:
       print("QUICK_TEST run: these numbers only prove the pipeline works. Set QUICK_TEST = False and re-run.")




if __name__ == "__main__":
   main()

