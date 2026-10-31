"""Step 3 - aligned CT/PET tumour crops (up to SLICES_PER_PATIENT annotated slices, CROP_FOV_MM field of view,
CROP_PX pixels, CT and PET on the same grid) plus simple tumour ROI features. Cached per patient.


Usage:  python 03_extract_crops.py


Input:  volumes + preprocess_log.csv (step 2), annotation_boxes.csv (step 1)
Output (CROP_DIR): <pid>.npz with ct, pet (n_slices x CROP_PX x CROP_PX, [0, 1]) and feat_json
"""
import os


import pandas as pd


from config import CROP_DIR, BOXES_CSV
from common import load_cohort, extract_patient_crops, load_crops




def main():
   cohort = load_cohort("volumes")
   boxes_df = pd.read_csv(BOXES_CSV)
   crop_fail = []
   todo = [r for _, r in cohort.iterrows() if not os.path.exists(f"{CROP_DIR}/{r['patient_id']}.npz")]
   print(f"Crop extraction: {len(cohort) - len(todo)} cached, {len(todo)} to build")
   for k, row in enumerate(todo):
       try:
           bp = boxes_df[(boxes_df["patient_id"] == row["patient_id"]) & (boxes_df["series_uid"] == row["annot_series_uid"])]
           extract_patient_crops(row["patient_id"], bp)
       except Exception as e:
           crop_fail.append((row["patient_id"], f"{type(e).__name__}: {e}")); print(f"  {row['patient_id']} FAILED {crop_fail[-1][1]}")
       if (k + 1) % 10 == 0:
           print(f"  crops {k + 1}/{len(todo)}")


   crops, _ = load_crops(cohort["patient_id"])
   cohort = cohort[cohort["patient_id"].isin(crops)].reset_index(drop=True)
   print(f"\nCrops ready for {len(cohort)} patients | total slices: {sum(v[0].shape[0] for v in crops.values())} | "
         f"label counts: {cohort['label'].value_counts().sort_index().to_dict()}")




if __name__ == "__main__":
   main()



