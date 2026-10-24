"""Step 3 - exploratory data analysis.


Output (FIG_ROOT/eda): class_balance.png, diameter_distribution.png, sample_crops.png, views_example.png,
pixel_intensity_hist.png
"""
import os


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


from config import PREP_DIR, FIG_ROOT


FIG_DIR = f"{FIG_ROOT}/eda"
sns.set_style("whitegrid")




def save(name):
   plt.tight_layout()
   plt.savefig(f"{FIG_DIR}/{name}", dpi=150)
   plt.close()




def main():
   os.makedirs(FIG_DIR, exist_ok=True)
   df = pd.read_csv(f"{PREP_DIR}/labels_split.csv")
   print(f"Total crops: {len(df)}")
   print(f"Unique patients/series: {df['series_uid'].nunique()}")
   print(df.groupby("split")["series_uid"].nunique(), "\n(patients per split)")


   # 1. Class balance overall and per split
   fig, axes = plt.subplots(1, 2, figsize=(12, 4))
   df["label"].value_counts().sort_index().plot(kind="bar", ax=axes[0], color=["#4C72B0", "#DD8452"])
   axes[0].set_title("Overall class balance")
   axes[0].set_xlabel("Label (0=non-nodule candidate, 1=nodule)")
   axes[0].set_ylabel("Count")
   split_counts = df.groupby(["split", "label"]).size().unstack(fill_value=0)
   split_counts.plot(kind="bar", stacked=True, ax=axes[1])
   axes[1].set_title("Class balance per split")
   axes[1].set_xlabel("Split")
   save("class_balance.png")


   # 2. Nodule diameter distribution (from annotations.csv; candidates have no diameter)
   nodules_csv = f"{PREP_DIR}/nodule_annotations.csv"
   if os.path.exists(nodules_csv):
       nodules = pd.read_csv(nodules_csv)
       nodules = nodules[nodules["seriesuid"].isin(df["series_uid"])]
       plt.figure(figsize=(7, 4))
       sns.histplot(nodules["diameter_mm"], bins=30, kde=True, color="#4C72B0")
       plt.title("Nodule diameter distribution (mm)")
       plt.xlabel("Diameter (mm)")
       save("diameter_distribution.png")
   else:
       print(f"No {nodules_csv} - rerun Step 1 (annotations.csv was not found in the dataset).")


   # 3. Sample crop grid (mix of both classes; axial channel shown)
   fig, axes = plt.subplots(2, 5, figsize=(15, 6))
   for label_val, row_idx in [(0, 0), (1, 1)]:
       subset = df[df["label"] == label_val].sample(min(5, (df["label"] == label_val).sum()), random_state=42)
       for col_idx, (_, row) in enumerate(subset.iterrows()):
           crop = np.load(os.path.join(PREP_DIR, row["filename"]))
           axes[row_idx, col_idx].imshow(crop[0], cmap="gray", vmin=0, vmax=1)
           axes[row_idx, col_idx].set_title(f"label={label_val}", fontsize=9)
           axes[row_idx, col_idx].axis("off")
   plt.suptitle("Sample nodule crops by class")
   save("sample_crops.png")


   # 3b. One nodule shown as the three input channels (illustrates the 2.5D representation)
   crop = np.load(os.path.join(PREP_DIR, df[df["label"] == 1].sample(1, random_state=42)["filename"].iloc[0]))
   fig, axes = plt.subplots(1, 3, figsize=(9, 3.3))
   for ax, view, title in zip(axes, crop, ["Axial (ch 0)", "Coronal (ch 1)", "Sagittal (ch 2)"]):
       ax.imshow(view, cmap="gray", vmin=0, vmax=1)
       ax.set_title(title, fontsize=10)
       ax.axis("off")
   plt.suptitle("2.5D input: three orthogonal 50 mm views of one nodule")
   save("views_example.png")


   # 4. Pixel intensity histogram (sanity check on normalization)
   sample_files = df["filename"].sample(min(200, len(df)), random_state=42)
   all_pixels = np.concatenate([np.load(os.path.join(PREP_DIR, f)).ravel() for f in sample_files])
   plt.figure(figsize=(7, 4))
   sns.histplot(all_pixels, bins=50, color="#55A868")
   plt.title("Pixel intensity distribution (post lung-window normalization)")
   plt.xlabel("Normalized intensity [0,1]")
   save("pixel_intensity_hist.png")


   print(f"\nAll EDA figures saved to {FIG_DIR}")




if __name__ == "__main__":
   main()
