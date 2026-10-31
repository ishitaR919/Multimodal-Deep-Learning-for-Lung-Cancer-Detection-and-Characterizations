"""Step 4 - EDA on the crops and the simple tumour ROI features.


Usage:  python 04_eda_crops.py


Input:  crops (step 3)
Output (FIG_DIR): eda3_crop_montage.png, eda4_features_by_class.png, eda5_feature_corr.png
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu


from config import PET_SUV_MAX, FIG_DIR, RAD_COLS
from common import load_state


sns.set_style("whitegrid")




def main():
   st = load_state()
   cohort, crops = st.cohort, st.crops
   fe = pd.DataFrame.from_dict(st.feats, orient="index").rename_axis("patient_id").reset_index()
   fe = fe.merge(cohort[["patient_id", "label", "subtype"]], on="patient_id")


   # --- montage: CT | PET | PET-on-CT overlay, 3 patients per class (middle slice) ---
   rows_plot = []
   for lab in (0, 1):
       rows_plot += cohort[cohort["label"] == lab]["patient_id"].head(3).tolist()
   fig, axes = plt.subplots(len(rows_plot), 3, figsize=(8, 2.7 * len(rows_plot)))
   axes = np.atleast_2d(axes)
   for i, pid in enumerate(rows_plot):
       ct, pet = crops[pid]; j = ct.shape[0] // 2
       axes[i, 0].imshow(ct[j], cmap="gray", vmin=0, vmax=1)
       axes[i, 1].imshow(pet[j] * PET_SUV_MAX, cmap="hot", vmin=0, vmax=PET_SUV_MAX)
       axes[i, 2].imshow(ct[j], cmap="gray", vmin=0, vmax=1)
       axes[i, 2].imshow(np.ma.masked_less(pet[j] * PET_SUV_MAX, 1.0), cmap="hot", alpha=0.5, vmin=1, vmax=PET_SUV_MAX)
       sub = cohort.loc[cohort.patient_id == pid, "subtype"].iloc[0]
       axes[i, 0].set_ylabel(f"{pid}\n{sub}", fontsize=8)
       for a in axes[i]:
           a.set_xticks([]); a.set_yticks([])
   for a, t in zip(axes[0], ["CT (lung window)", "PET (SUV)", "PET on CT"]):
       a.set_title(t, fontsize=9)
   plt.tight_layout(); plt.savefig(f"{FIG_DIR}/eda3_crop_montage.png", dpi=140); plt.close()


   # --- do the simple tumor features separate the classes? (Mann-Whitney U) ---
   feat_cols = RAD_COLS
   fig, axes = plt.subplots(2, 4, figsize=(17, 8))
   for ax, col in zip(axes.ravel(), feat_cols):
       sns.boxplot(data=fe, x="label", y=col, ax=ax, palette=["#4C72B0", "#DD8452"], showfliers=False)
       sns.stripplot(data=fe, x="label", y=col, ax=ax, color="k", size=3, alpha=0.5)
       a, b = fe.loc[fe.label == 0, col].dropna(), fe.loc[fe.label == 1, col].dropna()
       p = mannwhitneyu(a, b).pvalue if len(a) > 1 and len(b) > 1 else np.nan
       ax.set_title(f"{col}  (p={p:.3g})", fontsize=10)
   plt.tight_layout(); plt.savefig(f"{FIG_DIR}/eda4_features_by_class.png", dpi=140); plt.close()


   plt.figure(figsize=(7, 5.5))
   sns.heatmap(fe[feat_cols + ["label"]].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, vmin=-1, vmax=1)
   plt.title("Feature correlation (incl. label)"); plt.tight_layout()
   plt.savefig(f"{FIG_DIR}/eda5_feature_corr.png", dpi=140); plt.close()
   print(f"EDA figures saved to {FIG_DIR}")




if __name__ == "__main__":
   main()