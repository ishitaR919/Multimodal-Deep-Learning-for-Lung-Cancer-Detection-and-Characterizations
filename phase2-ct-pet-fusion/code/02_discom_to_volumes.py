"""Step 2 - convert each cohort patient's CT (HU) and PET (body-weight SUV) DICOM series to NIfTI volumes.
Resumable: patients already converted are skipped; one failed patient never stops the run.


Usage:  python 02_dicom_to_volumes.py


Input:  paired_patients_v2.csv (step 1) + DICOM folders
Output (VOL_DIR): <pid>_ct.nii.gz, <pid>_pet_suv.nii.gz
Output (RES_DIR): preprocess_log.csv (status, SUV method, SUVmax, CT/PET coverage, ...)
"""
import multiprocessing as mp
import os
import time


import pandas as pd
import SimpleITK as sitk


from config import PREP_WORKERS, LOG_CSV, VOL_DIR
from common import load_cohort, _safe_process_patient




def run_preprocessing(rows):
   records = {}
   if os.path.exists(LOG_CSV):
       for r in pd.read_csv(LOG_CSV).to_dict("records"):
           pid = r["patient_id"]
           if r.get("status") == "ok" and os.path.exists(f"{VOL_DIR}/{pid}_ct.nii.gz") and os.path.exists(f"{VOL_DIR}/{pid}_pet_suv.nii.gz"):
               records[pid] = r
   todo = [r for _, r in rows.iterrows() if r["patient_id"] not in records]
   print(f"DICOM conversion: {len(records)} already done, {len(todo)} to process with {PREP_WORKERS} workers")
   sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)          # one thread per worker process
   t0 = time.time()
   with mp.get_context("fork").Pool(PREP_WORKERS) as pool:
       for k, rec in enumerate(pool.imap_unordered(_safe_process_patient, todo), 1):
           if rec["status"] == "ok":
               print(f"  [{k}/{len(todo)}] {rec['patient_id']} ok  SUVmax={rec['suv_max']:.1f}  coverage={rec['ct_pet_coverage']}  ({rec['seconds']}s)"
                     f" | ~{(time.time() - t0) / k * (len(todo) - k) / 60:.0f} min left")
           else:
               print(f"  [{k}/{len(todo)}] {rec['patient_id']} FAILED  {rec['error']}")
           records[rec["patient_id"]] = rec
           pd.DataFrame(list(records.values())).to_csv(LOG_CSV, index=False)   # resumable after a restart
   sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(os.cpu_count() or 1)
   return pd.DataFrame(list(records.values()))




def main():
   cohort = load_cohort("labels")
   prep_log = run_preprocessing(cohort)
   prep_log = prep_log[prep_log["patient_id"].isin(cohort["patient_id"])]
   bad = prep_log[prep_log["status"] != "ok"]
   if len(bad):
       print("\nFailed patients (dropped from the cohort):"); print(bad[["patient_id", "error"]].to_string(index=False))
   cohort = cohort[cohort["patient_id"].isin(prep_log.loc[prep_log["status"] == "ok", "patient_id"])].reset_index(drop=True)
   ok_log = prep_log[prep_log["status"] == "ok"]
   print(f"\nCohort after DICOM conversion: {len(cohort)} patients")
   print(f"SUVmax median {ok_log['suv_max'].median():.1f} (min {ok_log['suv_max'].min():.1f}, max {ok_log['suv_max'].max():.1f}) | "
         f"suspicious (<1 or >60): {((ok_log['suv_max'] < 1) | (ok_log['suv_max'] > 60)).sum()}")
   print(f"Same FrameOfReferenceUID: {int(ok_log['same_frame_of_reference'].sum())}/{len(ok_log)} | "
         f"CT coverage by PET < 90%: {(ok_log['ct_pet_coverage'] < 0.9).sum()}")
   print("PET units / SUV method:", ok_log[["pet_units", "suv_method"]].value_counts().to_dict())




if __name__ == "__main__":
   main()