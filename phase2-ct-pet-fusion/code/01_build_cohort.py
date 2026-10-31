"""Step 1 - load the dataset tables, pair CT + PET from the same study, check which CT series holds the tumour
annotations, apply the fusion rule and define the labelled cohort.


Usage:  python 01_build_cohort.py


Input:  LUNG-PET-CT-Dx under INPUT_DIR (metadata.csv, statistics-clinical-20201221.xlsx, Annotation/, DICOM folders)
Output (RES_DIR): paired_patients_v2.csv, annotation_boxes.csv, annotation_patient_summary.csv
Output (FIG_DIR): eda1_cohort_overview.png, eda2_tumor_extent.png
"""
import os
import time
from concurrent.futures import ThreadPoolExecutor


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


from config import (QUICK_TEST, QUICK_N_PATIENTS, TASK, SEED, RECOMPUTE_ANNOTATIONS, MIN_MATCHED_SLICES, ANN_THREADS,
                   FIG_DIR, V2_CSV, BOXES_CSV, SUMM_CSV)
from common import (find_root, build_series_table, load_clinical, pair_patients, add_subtype, check_patient,
                   finalize_annotation_check, define_cohort)


sns.set_style("whitegrid")




def main():
   # ---- dataset tables and CT+PET pairing ----
   root = find_root()
   ann_root = os.path.join(root, "Annotation", "Annotation")
   series = build_series_table(root)
   clin = load_clinical(root)
   print(f"Dataset root: {root}")
   print(f"Series: {len(series)} | folders found: {series['dir_exists'].sum()} | missing: {(~series['dir_exists']).sum()}")


   paired = pair_patients(series)
   paired = add_subtype(paired).merge(clin, on="patient_id", how="left", validate="one_to_one")
   all_patients = add_subtype(pd.DataFrame({"patient_id": sorted(series["patient_id"].unique())}))
   ct_p, pt_p = set(series.loc[series.modality == "CT", "patient_id"]), set(series.loc[series.modality == "PT", "patient_id"])
   print(f"\nPatients: {series['patient_id'].nunique()} | with CT: {len(ct_p)} | with PET: {len(pt_p)} | "
         f"CT+PET same study: {len(paired)}")
   print("PET series types:", series.loc[series.modality == "PT", "series_desc"].value_counts().to_dict())
   counts = pd.DataFrame({"all_patients": all_patients["subtype"].value_counts(),
                          "ct_pet_paired": paired["subtype"].value_counts()}).fillna(0).astype(int)
   print("\nSubtype counts:\n", counts)
   print("\nClinical missing values (paired cohort):")
   print(paired[["sex", "age", "weight_kg", "t_stage", "n_stage", "m_stage", "grade", "smoking"]].isna().sum().to_string())


   # ---- EDA 1 ----
   fig, axes = plt.subplots(2, 3, figsize=(17, 9))
   counts.plot(kind="bar", ax=axes[0, 0], color=["#4C72B0", "#DD8452"]); axes[0, 0].set_title("Subtype counts")
   axes[0, 0].tick_params(axis="x", rotation=20)
   n_ct = series[series.modality == "CT"].groupby("patient_id").size()
   axes[0, 1].hist(n_ct, bins=range(1, int(n_ct.max()) + 2), color="#55A868"); axes[0, 1].set_title("CT series per patient (all 355)")
   axes[0, 2].hist(paired["ct_n_images"], bins=30, alpha=0.7, label="CT slices"); axes[0, 2].hist(paired["pt_n_images"], bins=30, alpha=0.7, label="PET slices")
   axes[0, 2].set_title("Slices per selected series (paired)"); axes[0, 2].legend()
   for sub in paired["subtype"].dropna().unique():
       axes[1, 0].hist(paired.loc[paired.subtype == sub, "age"].dropna(), bins=12, alpha=0.55, label=sub)
   axes[1, 0].set_title("Age by subtype"); axes[1, 0].legend()
   pd.crosstab(paired["subtype"], paired["smoking"]).plot(kind="bar", ax=axes[1, 1]); axes[1, 1].set_title("Smoking history by subtype")
   axes[1, 1].tick_params(axis="x", rotation=20)
   paired["t_stage"].astype(str).value_counts().sort_index().plot(kind="bar", ax=axes[1, 2], color="#8172B3"); axes[1, 2].set_title("T-stage counts (paired)")
   plt.tight_layout(); plt.savefig(f"{FIG_DIR}/eda1_cohort_overview.png", dpi=140); plt.close()


   # ---- annotation check (cached) ----
   if not RECOMPUTE_ANNOTATIONS and all(os.path.exists(p) for p in (BOXES_CSV, SUMM_CSV)):
       print("Using cached annotation check (set RECOMPUTE_ANNOTATIONS=True to redo).")
       boxes_df, summ_df = pd.read_csv(BOXES_CSV), pd.read_csv(SUMM_CSV)
   else:
       ct_series = series[series["modality"] == "CT"]
       series_by_uid = series.set_index("series_uid")
       to_check = paired
       if QUICK_TEST:                             # ~3x the quick cohort, stratified by subtype, is enough to sample from
           to_check = (paired.groupby("subtype", group_keys=False)
                       .apply(lambda g: g.sample(min(len(g), max(3, round(3 * QUICK_N_PATIENTS * len(g) / len(paired)))),
                                                 random_state=SEED)))
       print(f"Annotation check for {len(to_check)} patients with {ANN_THREADS} threads")
       all_boxes, summaries, t0 = [], [], time.time()
       with ThreadPoolExecutor(ANN_THREADS) as ex:
           rows = [r for _, r in to_check.iterrows()]
           for k, (boxes, summ) in enumerate(ex.map(lambda r: check_patient(r, ann_root, ct_series, series_by_uid), rows), 1):
               if not boxes.empty:
                   all_boxes.append(boxes)
               summaries.append(summ)
               if k % 10 == 0 or k == len(to_check):
                   el = time.time() - t0
                   print(f"  annotation check {k}/{len(to_check)} | {el / 60:.1f} min | ~{el / k * (len(to_check) - k) / 60:.1f} min left")


       boxes_df = pd.concat(all_boxes, ignore_index=True)
       summ_df = pd.DataFrame(summaries).merge(paired[["patient_id", "subtype_letter", "subtype", "study_uid"]], on="patient_id", how="left")
       summ_df["matches_selected_ct"] = summ_df["annot_series_uid"] == summ_df["selected_ct_series_uid"]
       summ_df["annot_in_pet_study"] = summ_df["annot_study_uid"] == summ_df["study_uid"]
       boxes_df.to_csv(BOXES_CSV, index=False); summ_df.to_csv(SUMM_CSV, index=False)


   v2, summ_df = finalize_annotation_check(paired, summ_df)
   v2.to_csv(V2_CSV, index=False)


   # ---- label check and EDA 2 ----
   print("\n=== Label check: patient-ID prefix vs XML class letter (rows = boxes) ===")
   _b = boxes_df.copy(); _b["id_prefix"] = _b["patient_id"].str[0].str.lower()
   print(pd.crosstab(_b["id_prefix"], _b["class_name"]))
   print(f"Patients with a box whose class letter != ID prefix: {_b[_b['id_prefix'] != _b['class_name']]['patient_id'].nunique()}")
   n = len(summ_df)
   print(f"\nShare of annotated slices found in one CT series: median {summ_df['match_frac'].median():.2f} "
         f"(quartiles {summ_df['match_frac'].quantile(0.25):.2f} / {summ_df['match_frac'].quantile(0.75):.2f})")
   print(f">= {MIN_MATCHED_SLICES} annotated slices found: {int((summ_df['n_matched'].fillna(0) >= MIN_MATCHED_SLICES).sum())}/{n} | "
         f"annotated CT is in the PET study: {int(summ_df['annot_in_pet_study'].sum())}/{n} | "
         f"usable (fusion_ok): {int(summ_df['fusion_ok'].sum())}/{n}")
   ok = summ_df[summ_df["fusion_ok"]]
   print(ok.groupby("subtype")["fusion_ok"].count().rename("usable_patients").to_string())
   fig, axes = plt.subplots(1, 3, figsize=(15, 4))
   axes[0].hist(ok["z_extent_mm"].dropna(), bins=25, color="#4C72B0"); axes[0].set_title("Tumor z-extent (mm)")
   axes[1].hist(ok["mean_box_w_px"].dropna(), bins=25, color="#DD8452"); axes[1].set_title("Mean box width (px)")
   axes[2].hist(ok["n_annotated_slices"].dropna(), bins=25, color="#55A868"); axes[2].set_title("Annotated slices per patient")
   plt.tight_layout(); plt.savefig(f"{FIG_DIR}/eda2_tumor_extent.png", dpi=140); plt.close()


   # ---- working cohort + labels ----
   cohort = define_cohort(v2)
   print(f"\nCOHORT: {len(cohort)} patients | label counts: {cohort['label'].value_counts().sort_index().to_dict()} "
         f"(1 = {'squamous/small cell/large cell' if TASK == 'adeno_vs_rest' else 'squamous'})")




if __name__ == "__main__":
   main()
