"""Step 2 - patient-level train / validation / test split (70 / 15 / 15).


All candidates of one scan go to the same split, so no patient appears in two splits.
Output: PREP_DIR/labels_split.csv
"""
import pandas as pd
from sklearn.model_selection import train_test_split


from config import PREP_DIR, RANDOM_SEED, TRAIN_FRAC, VAL_FRAC




def main():
   df = pd.read_csv(f"{PREP_DIR}/labels.csv")


   unique_patients = df["series_uid"].unique()
   print(f"Total unique patients/series: {len(unique_patients)}")


   train_patients, temp_patients = train_test_split(
       unique_patients, train_size=TRAIN_FRAC, random_state=RANDOM_SEED
   )
   val_size_relative = VAL_FRAC / (1 - TRAIN_FRAC)
   val_patients, test_patients = train_test_split(
       temp_patients, train_size=val_size_relative, random_state=RANDOM_SEED
   )


   df["split"] = "test"
   df.loc[df["series_uid"].isin(train_patients), "split"] = "train"
   df.loc[df["series_uid"].isin(val_patients), "split"] = "val"


   # sanity check: no patient overlap across splits
   train_set = set(df[df["split"] == "train"]["series_uid"])
   val_set = set(df[df["split"] == "val"]["series_uid"])
   test_set = set(df[df["split"] == "test"]["series_uid"])
   assert not (train_set & val_set), "Patient leakage between train/val!"
   assert not (train_set & test_set), "Patient leakage between train/test!"
   assert not (val_set & test_set), "Patient leakage between val/test!"
   print("No patient leakage across splits - confirmed.")


   print(df.groupby("split")["label"].agg(crops="count", positives="sum"))
   df.to_csv(f"{PREP_DIR}/labels_split.csv", index=False)
   print(f"Saved split file to {PREP_DIR}/labels_split.csv")

if __name__ == "__main__":
   main()
