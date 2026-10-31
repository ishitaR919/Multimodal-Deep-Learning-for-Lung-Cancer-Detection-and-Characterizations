"""Step 3 - EDA: Kaplan-Meier by stage, follow-up time, tumour volume by label, example crops with the GTV contour.


Usage:   python 03_eda.py
Inputs:  RES_DIR/clinical.csv, RES_DIR/cohort.csv, CROP_DIR/*.npz
Outputs: FIG_DIR/eda1_survival_stage_volume.png, FIG_DIR/eda2_crop_montage.png
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402
from scipy.stats import mannwhitneyu  # noqa: E402


from config import CLINICAL_CSV_OUT, FIG_DIR, HORIZON_DAYS  # noqa: E402
from common import load_state, km_curve  # noqa: E402


sns.set_style("whitegrid")




def main():
   st = load_state()
   clin = pd.read_csv(CLINICAL_CSV_OUT, dtype={"patient_id": str})
   cohort, VOL_COL = st.cohort, st.vol_col
   fe = pd.DataFrame.from_dict(st.feats, orient="index").rename_axis("patient_id").reset_index()
   fe = fe.merge(cohort[["patient_id", "label", "overall_stage", "time_days", "event"]], on="patient_id")


   fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
   for s, g in clin.dropna(subset=["time_days", "event"]).groupby(clin["overall_stage"].astype(str)):
       x, y = km_curve(g["time_days"], g["event"])
       axes[0].step(x / 365.25, y, where="post", label=f"{s} (n={len(g)})")
   axes[0].axvline(HORIZON_DAYS / 365.25, color="k", ls=":"); axes[0].set_xlabel("years"); axes[0].set_ylabel("overall survival")
   axes[0].set_title("Kaplan-Meier by overall stage (all clinical patients)"); axes[0].legend(fontsize=8)
   axes[1].hist(clin["time_days"].dropna() / 365.25, bins=40, color="#4C72B0"); axes[1].axvline(HORIZON_DAYS / 365.25, color="k", ls=":")
   axes[1].set_title("Survival / follow-up time (years)")
   sns.boxplot(data=fe, x="label", y=fe[VOL_COL] / 1000, ax=axes[2], palette=["#4C72B0", "#DD8452"], showfliers=False)
   sns.stripplot(data=fe, x="label", y=fe[VOL_COL] / 1000, ax=axes[2], color="k", size=2, alpha=0.4)
   p = mannwhitneyu(fe.loc[fe.label == 0, VOL_COL], fe.loc[fe.label == 1, VOL_COL]).pvalue if fe.label.nunique() > 1 else np.nan
   axes[2].set_yscale("log"); axes[2].set_ylabel("GTV volume (mL)"); axes[2].set_title(f"Tumour volume by label (p={p:.3g})")
   plt.tight_layout(); plt.savefig(f"{FIG_DIR}/eda1_survival_stage_volume.png", dpi=140); plt.close()


   rows_plot = [p for lab in (0, 1) for p in cohort[cohort["label"] == lab]["patient_id"].head(3)]
   fig, axes = plt.subplots(len(rows_plot), 3, figsize=(8, 2.7 * len(rows_plot)))
   axes = np.atleast_2d(axes)
   for i, pid in enumerate(rows_plot):
       ct, m = st.crops[pid]; j = ct.shape[0] // 2
       for v in range(3):
           axes[i, v].imshow(ct[j, v], cmap="gray", vmin=0, vmax=1)
           axes[i, v].contour(m[j, v], levels=[0.5], colors="r", linewidths=0.8)
           axes[i, v].set_xticks([]); axes[i, v].set_yticks([])
       axes[i, 0].set_ylabel(f"{pid}\nlabel={st.labels[pid]}", fontsize=8)
   for a, t in zip(axes[0], ["Axial", "Coronal", "Sagittal"]):
       a.set_title(t, fontsize=9)
   plt.suptitle("2.5D CT crops (128 mm) with GTV contour"); plt.tight_layout()
   plt.savefig(f"{FIG_DIR}/eda2_crop_montage.png", dpi=140); plt.close()


   print("Clinical missing values (cohort):",
         cohort[["age", "gender", "t_stage", "n_stage", "m_stage", "overall_stage", "histology"]].isna().sum().to_dict())
   print(f"EDA figures saved to {FIG_DIR}")




if __name__ == "__main__":
   main()