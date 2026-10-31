"""Step 1 - clinical table + DICOM index -> labelled cohort with a paired CT and RTSTRUCT per patient.


Label (2-year overall survival): 1 = died within HORIZON_DAYS, 0 = alive after HORIZON_DAYS.
Patients censored before 2 years have no binary label and are left out.


Usage:   python 01_build_cohort.py
Inputs:  CLINICAL_FILE (or a clinical CSV/XLSX found under INPUT_DIR), DICOM under IMAGE_ROOT
Outputs: RES_DIR/clinical.csv (all clinical patients), RES_DIR/dicom_index.csv (cached), RES_DIR/cohort.csv
"""
import os
import time


import pandas as pd


from config import (INPUT_DIR, IMAGE_ROOT, CLINICAL_FILE, INDEX_CSV, CLINICAL_CSV_OUT, COHORT_CSV, HORIZON_DAYS,
                   QUICK_TEST, QUICK_N_PATIENTS, SEED)
from common import find_clinical_csv, load_clinical, index_dicom, pair_series, make_label




def main():
   clin_path = find_clinical_csv()
   if os.path.isdir(IMAGE_ROOT):
       data_root = IMAGE_ROOT
   elif CLINICAL_FILE:                                 # separate clinical upload -> search all inputs for DICOM
       data_root = INPUT_DIR
   else:
       rel = os.path.relpath(clin_path, INPUT_DIR).split(os.sep)
       data_root = os.path.join(INPUT_DIR, *(rel[:3] if rel[0] == "datasets" else rel[:1]))
   print(f"Clinical file: {clin_path}\nDICOM root: {data_root}")
   clin = load_clinical(clin_path)
   clin.to_csv(CLINICAL_CSV_OUT, index=False)


   if os.path.exists(INDEX_CSV):
       dicom_idx = pd.read_csv(INDEX_CSV, dtype={"patient_id": str})
   else:
       t0 = time.time()
       dicom_idx = index_dicom(data_root)
       dicom_idx.to_csv(INDEX_CSV, index=False)
       print(f"Indexed {len(dicom_idx)} series in {time.time() - t0:.0f}s")
   if dicom_idx.empty:
       raise RuntimeError("No DICOM series found - check IMAGE_ROOT.")
   print("Series per modality:", dicom_idx["modality"].value_counts().to_dict())


   cohort = clin.merge(pair_series(dicom_idx), on="patient_id", how="inner")
   cohort["label"] = [make_label(t, e) for t, e in zip(cohort["time_days"], cohort["event"])]
   print(f"\nClinical patients: {len(clin)} | with CT + RTSTRUCT: {len(cohort)} | "
         f"RTSTRUCT references the chosen CT: {int(cohort['rt_matches_ct'].sum())}/{len(cohort)}")
   print(f"Censored before {HORIZON_DAYS} days (no binary label): {int(cohort['label'].isna().sum())}")
   cohort = cohort[cohort["label"].notna() & cohort["time_days"].notna()].reset_index(drop=True)
   cohort["label"] = cohort["label"].astype(int)
   if QUICK_TEST and len(cohort) > QUICK_N_PATIENTS:
       cohort = (cohort.groupby("label", group_keys=False)
                 .apply(lambda g: g.sample(max(5, round(QUICK_N_PATIENTS * len(g) / len(cohort))), random_state=SEED))
                 .sort_values("patient_id").reset_index(drop=True))
   cohort.to_csv(COHORT_CSV, index=False)
   print(f"COHORT: {len(cohort)} patients | label counts {cohort['label'].value_counts().sort_index().to_dict()} "
         f"(1 = died within 2 years) -> {COHORT_CSV}")




if __name__ == "__main__":
   main()