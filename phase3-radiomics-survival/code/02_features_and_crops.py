"""Step 2 - per patient: CT + GTV contour -> 1 mm isotropic tumour cube -> radiomic features + 2.5D crops.


Runs in parallel worker processes and caches one .npz per patient, so a restarted run only processes
missing patients.


Usage:   python 02_features_and_crops.py
Inputs:  RES_DIR/cohort.csv (step 1), CT + RTSTRUCT DICOM
Outputs: CROP_DIR/<patient_id>.npz (ct crops, mask crops, feature JSON), RES_DIR/preprocess_log.csv
"""
import multiprocessing as mp
import os
import time


import pandas as pd
import SimpleITK as sitk


from config import COHORT_CSV, PREP_LOG_CSV, N_WORKERS, FEATURE_SET
from common import process_patient, load_crops




def main():
   cohort = pd.read_csv(COHORT_CSV, dtype={"patient_id": str})
   jobs = cohort[["patient_id", "ct_dir", "ct_series_uid", "rt_file"]].to_dict("records")
   print(f"Preprocessing {len(jobs)} patients with {N_WORKERS} workers ({FEATURE_SET} features); "
         f"progress every 5 patients", flush=True)
   prep_log, t0 = [], time.time()
   sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
   with mp.get_context("fork").Pool(N_WORKERS) as pool:
       for k, rec in enumerate(pool.imap_unordered(process_patient, jobs), 1):
           prep_log.append(rec)
           if rec["status"] != "ok":
               print(f"  {rec['patient_id']} FAILED {rec['error']}", flush=True)
           if k % 5 == 0 or k == len(jobs):
               el = time.time() - t0
               print(f"  {k}/{len(jobs)} | {el / 60:.1f} min | ~{el / k * (len(jobs) - k) / 60:.1f} min left", flush=True)
   sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(os.cpu_count() or 1)
   prep_log = pd.DataFrame(prep_log)
   prep_log.to_csv(PREP_LOG_CSV, index=False)


   crops, feats = load_crops(cohort["patient_id"])
   print(f"\nReady: {len(crops)} patients | failed: {(prep_log['status'] != 'ok').sum()} | "
         f"radiomic features per patient: {len(next(iter(feats.values()))) if feats else 0}")




if __name__ == "__main__":
   main()
