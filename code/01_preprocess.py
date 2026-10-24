"""Step 1 - LUNA16 candidates -> balanced set of 2.5D crops.


Task = LUNA16 false-positive reduction: label 1 = candidate is a real nodule, 0 = not a nodule (NOT malignancy).
Each candidate becomes a 2.5D patch: axial / coronal / sagittal planes through the candidate centre, stacked as
3 channels (Roth et al. 2014; Setio et al. 2016 used 50 mm patches at 64 px).


Output: PREP_DIR/<seriesuid>_<idx>.npy (3, 64, 64) in [0, 1], PREP_DIR/labels.csv, PREP_DIR/nodule_annotations.csv
"""
import os
import shutil
from pathlib import Path


import numpy as np
import pandas as pd
import SimpleITK as sitk


from config import (INPUT_DIR, PREP_DIR, CROP_SIZE, CROP_MM, HU_MIN, HU_MAX,
                   NEGATIVE_TO_POSITIVE_RATIO, RANDOM_SEED)




def find_file(name, search_dir=INPUT_DIR):
   for root, dirs, files in os.walk(search_dir):
       if name in files:
           return os.path.join(root, name)
   return None




def load_ct_volume(mhd_path):
   """Load a .mhd/.raw CT scan, clip to the lung window and scale to [0, 1].
   Kept as a SimpleITK image so origin, spacing and direction stay attached to the voxels."""
   img = sitk.ReadImage(mhd_path, sitk.sitkFloat32)
   img = sitk.Clamp(img, sitk.sitkFloat32, HU_MIN, HU_MAX)
   return (img - HU_MIN) / (HU_MAX - HU_MIN)




def extract_25d_crop(img, world_xyz, crop_size=CROP_SIZE, crop_mm=CROP_MM):
   """Resample an isotropic cube of crop_mm around the candidate (world mm coordinates) and return its three
   central orthogonal planes (axial, coronal, sagittal) as a (3, crop_size, crop_size) array.
   Resampling to a fixed mm grid removes the scanner-dependent pixel spacing / slice thickness."""
   sp = crop_mm / crop_size
   D = np.array(img.GetDirection()).reshape(3, 3)
   ref = sitk.Image(crop_size, crop_size, crop_size, sitk.sitkFloat32)
   ref.SetSpacing((sp, sp, sp))
   ref.SetDirection(img.GetDirection())
   ref.SetOrigin(tuple(float(v) for v in np.asarray(world_xyz) - D @ np.full(3, sp * (crop_size - 1) / 2.0)))
   # outside the scan -> 0 (= -1000 HU, air)
   cube = sitk.GetArrayFromImage(sitk.Resample(img, ref, sitk.Transform(), sitk.sitkLinear, 0.0))  # z, y, x
   c = crop_size // 2
   return np.stack([cube[c], cube[:, c, :], cube[:, :, c]]).astype(np.float32)




def main():
   # candidates_V2.csv is the version recommended by the LUNA16 organisers; fall back to V1 if the mirror lacks it.
   candidates_csv = find_file("candidates_V2.csv") or find_file("candidates.csv")
   if candidates_csv is None:
       raise FileNotFoundError(f"No candidates(_V2).csv under {INPUT_DIR}. Check that the LUNA16 dataset is attached.")
   nodule_annotations_csv = find_file("annotations.csv")     # nodule centres + diameters (used for EDA only)
   print(f"Candidates file: {candidates_csv}")
   os.makedirs(PREP_DIR, exist_ok=True)


   annotations = pd.read_csv(candidates_csv)
   print(f"Loaded {len(annotations)} total candidates")


   # one path per series (some Kaggle mirrors contain duplicated subset folders)
   mhd_files = {p.stem: p for p in Path(INPUT_DIR).rglob("*.mhd")}
   print(f"Found {len(mhd_files)} CT series")
   annotations = annotations[annotations["seriesuid"].isin(mhd_files)]
   print(annotations["class"].value_counts(), "\n(class balance before subsampling, available scans only)")


   # Balance: keep all positives, subsample negatives
   positives = annotations[annotations["class"] == 1]
   negatives = annotations[annotations["class"] == 0]
   n_negatives_to_keep = min(len(negatives), len(positives) * NEGATIVE_TO_POSITIVE_RATIO)
   negatives_sampled = negatives.sample(n=n_negatives_to_keep, random_state=RANDOM_SEED)
   annotations = pd.concat([positives, negatives_sampled]).sample(frac=1, random_state=RANDOM_SEED)
   print(f"After subsampling: {len(positives)} positive, {len(negatives_sampled)} negative "
         f"({len(annotations)} total candidates to process)")


   records = []
   for series_uid, series_annotations in annotations.groupby("seriesuid"):
       img = load_ct_volume(str(mhd_files[series_uid]))


       for idx, row in series_annotations.iterrows():
           crop = extract_25d_crop(img, (row["coordX"], row["coordY"], row["coordZ"]))


           out_name = f"{series_uid}_{idx}.npy"
           np.save(os.path.join(PREP_DIR, out_name), crop)


           records.append({
               "filename": out_name,
               "series_uid": series_uid,
               "label": int(row["class"]),
           })


   labels_df = pd.DataFrame(records)
   labels_df.to_csv(os.path.join(PREP_DIR, "labels.csv"), index=False)
   if nodule_annotations_csv:
       shutil.copy(nodule_annotations_csv, os.path.join(PREP_DIR, "nodule_annotations.csv"))
   print(f"Saved {len(labels_df)} preprocessed crops to {PREP_DIR}")
   if len(labels_df) > 0 and "label" in labels_df.columns:
       print(labels_df["label"].value_counts())
   else:
       print("WARNING: no crops were produced. 0 matching CT series means the .mhd files were not found "
             f"under {INPUT_DIR} or their names do not match candidates seriesuid.")




if __name__ == "__main__":
   main()