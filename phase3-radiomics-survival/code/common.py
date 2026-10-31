"""Data, feature, model and evaluation code shared by the Phase 3 scripts.


Scripts never set globals in this module. Everything that depends on the cohort (crops, labels, folds,
tabular features) lives in a `State` object built from files on disk by `load_state()` and is passed in
explicitly.
"""
import json
import os
import random
import re


import numpy as np
import pandas as pd


from config import (INPUT_DIR, CLINICAL_FILE, HORIZON_DAYS, INDEX_THREADS, BIN_WIDTH, SLICES_PER_PATIENT, CROP_PX,
                   CT_WIN, HAVE_PYRADIOMICS, CROP_DIR, K_RAD, SEED, N_FOLDS, COHORT_CSV, RES_DIR, BATCH_SIZE, LR,
                   EPOCHS, MOD_DROP, WEIGHT_DIR, SAVE_WEIGHTS)




# =============================================================== clinical table + DICOM index
def _norm(c):
   return re.sub(r"[^a-z0-9]", "", str(c).lower())




def _read_table(path, nrows=None):
   return pd.read_excel(path, nrows=nrows) if path.lower().endswith((".xlsx", ".xls")) else pd.read_csv(path, nrows=nrows)




def _is_clinical(cols):
   """Lung1 clinical table: a survival-time column and a dead-status / event column."""
   has_time = any("survival" in c or c in ("time", "survivaltime", "ostime") for c in cols)
   has_event = any("dead" in c or "event" in c or "vitalstatus" in c for c in cols)
   return has_time and has_event




def find_clinical_csv(search_dir=INPUT_DIR, max_depth=8):
   """Walk for the Lung1 clinical table (.csv / .xlsx). DICOM-only folders are skipped quickly."""
   if CLINICAL_FILE:
       if not os.path.exists(CLINICAL_FILE):
           raise FileNotFoundError(f"CLINICAL_FILE not found: {CLINICAL_FILE} - is the clinical dataset attached?")
       return CLINICAL_FILE
   base, tables = search_dir.rstrip("/").count(os.sep), []
   for root, dirs, files in os.walk(search_dir):
       if root.count(os.sep) - base >= max_depth or any(f.lower().endswith(".dcm") for f in files[:5]):
           dirs[:] = []                       # stop below depth limit and inside DICOM series folders
       for f in files:
           if f.lower().endswith((".csv", ".xlsx", ".xls")):
               p = os.path.join(root, f)
               tables.append(p)
               try:
                   if _is_clinical([_norm(c) for c in _read_table(p, nrows=0).columns]):
                       return p
               except Exception:
                   continue
   print("Tabular files found:" if tables else f"No CSV/XLSX files at all under {search_dir}.")
   for p in tables[:30]:
       try:
           print(" ", p, "->", list(_read_table(p, nrows=0).columns)[:12])
       except Exception as e:
           print(" ", p, "-> unreadable:", type(e).__name__)
   raise FileNotFoundError(
       "No clinical table with survival-time and dead-status columns was found. Set CLINICAL_FILE to its path, or "
       "download the 'NSCLC-Radiomics-Lung1.clinical' CSV from TCIA (collection NSCLC-Radiomics).")




def load_clinical(path):
   raw = _read_table(path)
   cols = {_norm(c): c for c in raw.columns}


   def pick(*keys):
       for k in keys:                        # exact name first ('age' must not match 'overall.stage')
           if k in cols:
               return raw[cols[k]]
       for k in keys:
           for n, c in cols.items():
               if k in n:
                   return raw[c]
       return pd.Series(np.nan, index=raw.index)


   return pd.DataFrame({
       "patient_id": pick("patientid").astype(str).str.strip(),
       "age": pd.to_numeric(pick("age"), errors="coerce"),
       "gender": pick("gender", "sex"),
       "t_stage": pick("tstage"), "n_stage": pick("nstage"), "m_stage": pick("mstage"),
       "overall_stage": pick("overallstage"), "histology": pick("histology"),
       "time_days": pd.to_numeric(pick("survivaltime", "survival", "ostime", "time"), errors="coerce"),
       "event": pd.to_numeric(pick("deadstatus", "dead", "event", "vitalstatus"), errors="coerce"),
   })




def _read_series_header(item):
   import pydicom
   d, dcm = item
   try:
       ds = pydicom.dcmread(os.path.join(d, dcm[0]), stop_before_pixels=True)
   except Exception:
       return None
   ref = None
   if ds.Modality == "RTSTRUCT":
       try:
           ref = str(ds.ReferencedFrameOfReferenceSequence[0].RTReferencedStudySequence[0]
                     .RTReferencedSeriesSequence[0].SeriesInstanceUID)
       except (AttributeError, IndexError):
           pass
   return {"patient_id": str(ds.PatientID).strip(), "modality": str(ds.Modality),
           "series_uid": str(ds.SeriesInstanceUID), "dir": d, "n_files": len(dcm),
           "first_file": os.path.join(d, dcm[0]), "ref_series_uid": ref}




def index_dicom(root):
   """One header per folder (TCIA layout: patient/study/series/*.dcm), read in parallel threads."""
   from concurrent.futures import ThreadPoolExecutor
   items = []
   for d, _, files in os.walk(root):
       dcm = sorted(f for f in files if f.lower().endswith(".dcm"))
       if dcm:
           items.append((d, dcm))
   print(f"Reading {len(items)} series headers with {INDEX_THREADS} threads", flush=True)
   with ThreadPoolExecutor(INDEX_THREADS) as ex:
       rows = [r for r in ex.map(_read_series_header, items) if r is not None]
   return pd.DataFrame(rows)




def pair_series(idx):
   """CT = the series the RTSTRUCT references (else the CT with most slices)."""
   rows = []
   for pid, g in idx.groupby("patient_id"):
       ct, rt = g[g.modality == "CT"], g[g.modality == "RTSTRUCT"]
       if ct.empty or rt.empty:
           continue
       r = rt.sort_values("first_file").iloc[0]
       hit = ct[ct.series_uid == r["ref_series_uid"]]
       c = (hit if len(hit) else ct.sort_values("n_files", ascending=False)).iloc[0]
       rows.append({"patient_id": pid, "ct_dir": c["dir"], "ct_series_uid": c["series_uid"], "ct_n_slices": c["n_files"],
                    "rt_file": r["first_file"], "rt_matches_ct": bool(len(hit))})
   return pd.DataFrame(rows)




def make_label(t, e):
   if e == 1 and t <= HORIZON_DAYS:
       return 1.0
   if t > HORIZON_DAYS:
       return 0.0
   return np.nan                              # censored before 2 years: outcome unknown




# =============================================================== CT + GTV -> features + crops
def read_ct(series_dir, series_uid):
   import SimpleITK as sitk
   r = sitk.ImageSeriesReader()
   r.SetFileNames(r.GetGDCMSeriesFileNames(series_dir, series_uid))
   return r.Execute()                                        # HU (rescale slope/intercept applied)




def choose_roi(names):
   """Primary tumour contour: GTV-1 (as in Aerts et al. 2014), else the first ROI whose name starts with GTV."""
   norm = {k: _norm(v) for k, v in names.items()}
   for k, n in norm.items():
       if n == "gtv1":
           return k
   gtv = [k for k, n in sorted(norm.items()) if n.startswith("gtv")]
   return gtv[0] if gtv else None




def rtstruct_mask(rt_file, ct):
   """Rasterise the chosen RTSTRUCT contour on the CT grid. Contours on the same slice are XOR-ed (holes)."""
   import pydicom
   from skimage.draw import polygon
   ds = pydicom.dcmread(rt_file)
   names = {int(r.ROINumber): str(r.ROIName) for r in ds.StructureSetROISequence}
   roi = choose_roi(names)
   if roi is None:
       raise ValueError(f"no GTV contour (ROIs: {sorted(names.values())})")
   rc = next(r for r in ds.ROIContourSequence if int(r.ReferencedROINumber) == roi)
   size = ct.GetSize()                                       # x, y, z
   M_inv = np.linalg.inv(np.array(ct.GetDirection()).reshape(3, 3) @ np.diag(ct.GetSpacing()))
   origin = np.array(ct.GetOrigin())
   mask = np.zeros(size[::-1], dtype=bool)                   # z, y, x
   for c in getattr(rc, "ContourSequence", []):
       pts = np.array(c.ContourData, dtype=float).reshape(-1, 3)
       if len(pts) < 3:
           continue
       ijk = (pts - origin) @ M_inv.T                        # continuous voxel index (x, y, z)
       k = int(round(ijk[:, 2].mean()))
       if 0 <= k < size[2]:
           rr, cc = polygon(ijk[:, 1], ijk[:, 0], shape=(size[1], size[0]))
           mask[k, rr, cc] ^= True
   return mask, names[roi]




def iso_cube(ct, mask, margin_mm=20.0):
   """Resample a cube around the tumour onto a 1 mm isotropic grid (IBSI-style), CT linear, mask nearest."""
   import SimpleITK as sitk
   zz, yy, xx = np.nonzero(mask)
   lo, hi = np.array([xx.min(), yy.min(), zz.min()]), np.array([xx.max(), yy.max(), zz.max()])
   centre = np.array(ct.TransformContinuousIndexToPhysicalPoint(tuple(float(v) for v in (lo + hi) / 2.0)))
   n = int(np.ceil(((hi - lo + 1) * np.array(ct.GetSpacing())).max() + 2 * margin_mm))
   D = np.array(ct.GetDirection()).reshape(3, 3)
   ref = sitk.Image(n, n, n, sitk.sitkFloat32)
   ref.SetSpacing((1.0, 1.0, 1.0)); ref.SetDirection(ct.GetDirection())
   ref.SetOrigin(tuple(float(v) for v in centre - D @ np.full(3, (n - 1) / 2.0)))
   m_img = sitk.GetImageFromArray(mask.astype(np.uint8)); m_img.CopyInformation(ct)
   cube = sitk.GetArrayFromImage(sitk.Resample(ct, ref, sitk.Transform(), sitk.sitkLinear, -1024.0, sitk.sitkFloat32))
   m = sitk.GetArrayFromImage(sitk.Resample(m_img, ref, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8))
   return cube, m.astype(bool)




def pyradiomics_features(cube, mask):
   import logging
   import SimpleITK as sitk
   import radiomics
   from radiomics import featureextractor
   radiomics.logger.setLevel(logging.ERROR)
   ext = featureextractor.RadiomicsFeatureExtractor(binWidth=BIN_WIDTH, label=1)
   ext.disableAllImageTypes(); ext.enableImageTypeByName("Original"); ext.enableAllFeatures()
   out = ext.execute(sitk.GetImageFromArray(cube), sitk.GetImageFromArray(mask.astype(np.uint8)))
   return {"rad_" + k[len("original_"):]: float(v) for k, v in out.items() if k.startswith("original_")}




def lite_features(cube, mask):
   """Fallback radiomics on the 1 mm grid: shape, first-order and GLCM (definitions follow pyradiomics / IBSI)."""
   from scipy.stats import skew, kurtosis
   from scipy.spatial import ConvexHull
   from scipy.spatial.distance import pdist
   from skimage.measure import marching_cubes, mesh_surface_area
   from skimage.feature import graycomatrix
   f = {}
   # shape
   vol = float(mask.sum())                                   # mm^3 (1 mm voxels)
   verts, faces, _, _ = marching_cubes(np.pad(mask, 1).astype(np.float32), 0.5)
   area = float(mesh_surface_area(verts, faces))
   ev = np.sort(np.linalg.eigvalsh(np.cov(np.argwhere(mask).T.astype(float))))[::-1]
   hull = verts[ConvexHull(verts).vertices]
   f.update({"rad_shape_VoxelVolume": vol, "rad_shape_SurfaceArea": area,
             "rad_shape_SurfaceVolumeRatio": area / vol,
             "rad_shape_Sphericity": (36 * np.pi * vol ** 2) ** (1 / 3) / area,
             "rad_shape_Maximum3DDiameter": float(pdist(hull).max()),
             "rad_shape_MajorAxisLength": 4 * np.sqrt(ev[0]), "rad_shape_MinorAxisLength": 4 * np.sqrt(ev[1]),
             "rad_shape_LeastAxisLength": 4 * np.sqrt(ev[2]),
             "rad_shape_Elongation": np.sqrt(ev[1] / ev[0]), "rad_shape_Flatness": np.sqrt(ev[2] / ev[0])})
   # first order
   v = cube[mask]
   q10, q25, q50, q75, q90 = np.percentile(v, [10, 25, 50, 75, 90])
   _, counts = np.unique(np.floor((v - v.min()) / BIN_WIDTH), return_counts=True)
   p = counts / counts.sum()
   f.update({"rad_firstorder_Mean": v.mean(), "rad_firstorder_StandardDeviation": v.std(),
             "rad_firstorder_Skewness": skew(v), "rad_firstorder_Kurtosis": kurtosis(v, fisher=False),
             "rad_firstorder_Median": q50, "rad_firstorder_10Percentile": q10, "rad_firstorder_90Percentile": q90,
             "rad_firstorder_InterquartileRange": q75 - q25, "rad_firstorder_Range": v.max() - v.min(),
             "rad_firstorder_Entropy": -(p * np.log2(p)).sum(), "rad_firstorder_Uniformity": (p ** 2).sum()})
   # GLCM: per axial slice, 4 directions, distance 1, matrices summed; level 0 = outside the mask (dropped)
   lv = np.where(mask, np.floor((cube - v.min()) / BIN_WIDTH).astype(int) + 1, 0)
   L = int(lv.max()) + 1
   P = np.zeros((L, L))
   for z in np.nonzero(mask.any(axis=(1, 2)))[0]:
       P += graycomatrix(lv[z].astype(np.uint16), [1], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                         levels=L, symmetric=True).sum(axis=(2, 3))
   P = P[1:, 1:]
   P = P / max(P.sum(), 1e-12)
   i, j = np.indices(P.shape) + 1
   mi, mj = (i * P).sum(), (j * P).sum()
   si, sj = np.sqrt(((i - mi) ** 2 * P).sum()), np.sqrt(((j - mj) ** 2 * P).sum())
   nz = P[P > 0]
   f.update({"rad_glcm_Contrast": ((i - j) ** 2 * P).sum(), "rad_glcm_DifferenceAverage": (np.abs(i - j) * P).sum(),
             "rad_glcm_Idm": (P / (1 + (i - j) ** 2)).sum(), "rad_glcm_JointEnergy": (P ** 2).sum(),
             "rad_glcm_JointEntropy": -(nz * np.log2(nz)).sum(),
             "rad_glcm_Correlation": ((i - mi) * (j - mj) * P).sum() / (si * sj) if si * sj > 0 else 1.0})
   return {k: float(x) for k, x in f.items()}




def tumour_crops(cube, mask, k=SLICES_PER_PATIENT, n=CROP_PX):
   """k 2.5D crops (axial / coronal / sagittal planes through the tumour centroid of k slices spread over the
   central 80% of the tumour). Returns CT in [0, 1] (float16) and the matching mask planes."""
   h = n // 2
   cp, mp_ = np.pad(cube, h, constant_values=-1024.0), np.pad(mask, h)
   zs = np.nonzero(mask.any(axis=(1, 2)))[0]
   lo, hi = zs[int(0.1 * (len(zs) - 1))], zs[int(round(0.9 * (len(zs) - 1)))]
   imgs, msks = [], []
   for z in np.unique(np.round(np.linspace(lo, hi, k)).astype(int)):
       yy, xx = np.nonzero(mask[z])
       Z, Y, X = z + h, int(round(yy.mean())) + h, int(round(xx.mean())) + h
       for arr, out in ((cp, imgs), (mp_, msks)):
           out.append(np.stack([arr[Z, Y - h:Y + h, X - h:X + h], arr[Z - h:Z + h, Y, X - h:X + h],
                                arr[Z - h:Z + h, Y - h:Y + h, X]]))
   ct = (np.clip(np.stack(imgs), *CT_WIN) - CT_WIN[0]) / (CT_WIN[1] - CT_WIN[0])
   return ct.astype(np.float16), np.stack(msks).astype(np.uint8)




def process_patient(job):
   """Worker: one patient -> CROP_DIR/<pid>.npz (CT crops, mask crops, radiomic features). Cached."""
   import time
   pid, out, t0 = job["patient_id"], f"{CROP_DIR}/{job['patient_id']}.npz", time.time()
   if os.path.exists(out):
       return {"patient_id": pid, "status": "ok", "cached": True}
   try:
       ct = read_ct(job["ct_dir"], job["ct_series_uid"])
       mask, roi = rtstruct_mask(job["rt_file"], ct)
       if mask.sum() < 10:
           raise ValueError("GTV contour is empty or does not overlap the CT")
       cube, m = iso_cube(ct, mask)
       feats = pyradiomics_features(cube, m) if HAVE_PYRADIOMICS else lite_features(cube, m)
       cts, ms = tumour_crops(cube, m)
       np.savez_compressed(out, ct=cts, mask=ms, feat_json=np.array(json.dumps(feats)))
       return {"patient_id": pid, "status": "ok", "roi": roi, "gtv_ml": m.sum() / 1000.0,
               "seconds": round(time.time() - t0, 1)}
   except Exception as e:
       return {"patient_id": pid, "status": "failed", "error": f"{type(e).__name__}: {e}"}




# =============================================================== survival statistics
def km_curve(t, e):
   t, e = np.asarray(t, float), np.asarray(e, int)
   times, s, surv = np.unique(t[e == 1]), 1.0, []
   for u in times:
       s *= 1 - ((t == u) & (e == 1)).sum() / (t >= u).sum()
       surv.append(s)
   return np.concatenate([[0.0], times]), np.concatenate([[1.0], surv])




def logrank_p(t, e, g):
   from scipy.stats import chi2
   t, e, g = np.asarray(t, float), np.asarray(e, int), np.asarray(g, int)
   o_e, var = 0.0, 0.0
   for u in np.unique(t[e == 1]):
       at = t >= u
       n, n1 = at.sum(), (at & (g == 1)).sum()
       d, d1 = ((t == u) & (e == 1)).sum(), ((t == u) & (e == 1) & (g == 1)).sum()
       o_e += d1 - d * n1 / n
       if n > 1:
           var += d * (n1 / n) * (1 - n1 / n) * (n - d) / (n - 1)
   return float(chi2.sf(o_e ** 2 / var, 1)) if var > 0 else np.nan




def c_index(t, e, risk):
   """Harrell's concordance index (higher risk should mean earlier death)."""
   t, e, r = np.asarray(t, float), np.asarray(e, int), np.asarray(risk, float)
   num = den = 0.0
   for i in np.nonzero(e == 1)[0]:
       m = t > t[i]
       den += m.sum(); num += (r[i] > r[m]).sum() + 0.5 * (r[i] == r[m]).sum()
   return num / den if den else np.nan




# =============================================================== master table, folds, tabular pipelines
from sklearn.base import BaseEstimator, TransformerMixin                          # noqa: E402
from sklearn.model_selection import StratifiedKFold, train_test_split               # noqa: E402
from sklearn.impute import SimpleImputer                                             # noqa: E402
from sklearn.preprocessing import StandardScaler                                     # noqa: E402
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif      # noqa: E402
from sklearn.compose import ColumnTransformer                                        # noqa: E402
from sklearn.pipeline import Pipeline                                                # noqa: E402
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, brier_score_loss  # noqa: E402


STAGE_MAP = {"i": 1.0, "ii": 2.0, "iiia": 3.0, "iiib": 3.5, "iv": 4.0}
CLIN_COLS = ["age", "sex_male", "t_num", "n_num", "m_num", "stage_num", "hist_adeno", "hist_squamous", "hist_large", "hist_nos"]




def vol_col(columns):
   return "rad_shape_VoxelVolume" if "rad_shape_VoxelVolume" in columns else "rad_shape_MeshVolume"




def build_master(cohort_df, feats):
   m = cohort_df[["patient_id", "label", "time_days", "event", "age", "gender", "t_stage", "n_stage", "m_stage",
                  "overall_stage", "histology"]].copy()
   m["sex_male"] = np.where(m["gender"].isna(), np.nan, (m["gender"].astype(str).str.lower().str[0] == "m").astype(float))
   for c in ("t_stage", "n_stage", "m_stage"):
       m[c.replace("_stage", "_num")] = pd.to_numeric(m[c], errors="coerce")
   m["stage_num"] = m["overall_stage"].astype(str).str.strip().str.lower().map(STAGE_MAP)
   h = m["histology"].astype(str).str.lower()
   for key, col in (("adeno", "hist_adeno"), ("squam", "hist_squamous"), ("large", "hist_large"), ("nos", "hist_nos")):
       m[col] = h.str.contains(key).astype(float)
   fe_ = pd.DataFrame.from_dict(feats, orient="index").rename_axis("patient_id").reset_index()
   m = m.merge(fe_, on="patient_id", how="left")
   m["log_gtv_volume"] = np.log1p(m[vol_col(m.columns)])
   return m




class CorrelationFilter(BaseEstimator, TransformerMixin):
   """Drop a feature if |Pearson r| >= threshold with an already kept one (radiomic features are very redundant)."""
   def __init__(self, threshold=0.9):
       self.threshold = threshold


   def fit(self, X, y=None):
       X = np.asarray(X, float)
       if X.shape[1] < 2:
           self.keep_ = np.arange(X.shape[1]); return self
       C = np.nan_to_num(np.abs(np.corrcoef(X, rowvar=False)))
       keep = []
       for j in range(X.shape[1]):
           if all(C[j, k] < self.threshold for k in keep):
               keep.append(j)
       self.keep_ = np.array(keep)
       return self


   def transform(self, X):
       return np.asarray(X, float)[:, self.keep_]




def clin_steps():
   return [("imp", SimpleImputer(strategy="median", keep_empty_features=True)), ("sc", StandardScaler())]




def rad_steps():
   """Fitted on the training patients of each fold only (no selection leakage)."""
   return [("imp", SimpleImputer(strategy="median", keep_empty_features=True)), ("var", VarianceThreshold(1e-8)),
           ("sc", StandardScaler()), ("corr", CorrelationFilter(0.9)), ("kbest", SelectKBest(f_classif, k=K_RAD))]




def tab_estimator(cols, clf):
   rad = [c for c in cols if c.startswith("rad_")]
   cl = [c for c in cols if not c.startswith("rad_")]
   parts = ([("clin", Pipeline(clin_steps()), cl)] if cl else []) + ([("rad", Pipeline(rad_steps()), rad)] if rad else [])
   return Pipeline([("prep", ColumnTransformer(parts)), ("clf", clf)])




class State:
   """Everything that depends on the cohort; rebuilt deterministically from disk by load_state()."""


   def __init__(self, cohort, crops, feats):
       self.cohort, self.crops, self.feats = cohort, crops, feats
       self.master = build_master(cohort, feats).sort_values("patient_id").reset_index(drop=True)
       self.vol_col = vol_col(self.master.columns)
       self.clin_cols = CLIN_COLS
       self.rad_cols = sorted(c for c in self.master.columns if c.startswith("rad_"))
       self.pids = self.master["patient_id"].tolist()
       self.labels = dict(zip(self.master["patient_id"], self.master["label"].astype(int)))
       self.time = dict(zip(self.master["patient_id"], self.master["time_days"]))
       self.event = dict(zip(self.master["patient_id"], self.master["event"].astype(int)))
       self.xtab = self.master.set_index("patient_id")
       skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
       self.fold_of = {}
       for f, (_, te) in enumerate(skf.split(self.pids, [self.labels[p] for p in self.pids])):
           for i in te:
               self.fold_of[self.pids[i]] = f




def load_crops(pids):
   crops, feats = {}, {}
   for pid in pids:
       pth = f"{CROP_DIR}/{pid}.npz"
       if os.path.exists(pth):
           z = np.load(pth)
           crops[pid] = (z["ct"].astype(np.float32), z["mask"])
           feats[pid] = json.loads(str(z["feat_json"]))
   return crops, feats




def load_state(verbose=True):
   """Cohort (01) restricted to patients whose crops exist (02) -> State."""
   if not os.path.exists(COHORT_CSV):
       raise FileNotFoundError(f"{COHORT_CSV} missing - run 01_build_cohort.py first.")
   cohort = pd.read_csv(COHORT_CSV, dtype={"patient_id": str})
   crops, feats = load_crops(cohort["patient_id"])
   if not crops:
       raise FileNotFoundError(f"No crops in {CROP_DIR} - run 02_features_and_crops.py first.")
   cohort = cohort[cohort["patient_id"].isin(crops)].reset_index(drop=True)
   st = State(cohort, crops, feats)
   if verbose:
       print(f"State: {len(st.pids)} patients | {len(st.clin_cols)} clinical, {len(st.rad_cols)} radiomic features | "
             f"label counts {st.master['label'].value_counts().sort_index().to_dict()}")
   return st




def split_train_val(st, fold):
   """Deterministic train / validation split inside the training folds (same split for training and Grad-CAM)."""
   trv = [p for p in st.pids if st.fold_of[p] != fold]
   try:
       return train_test_split(trv, test_size=0.15, stratify=[st.labels[p] for p in trv], random_state=SEED + fold)
   except ValueError:
       return train_test_split(trv, test_size=0.15, random_state=SEED + fold)




def fit_vectors(st, tr, use_rad, use_clin):
   """Tabular inputs for the neural models: pipelines fitted on training patients, applied to everyone."""
   y = [st.labels[p] for p in tr]
   out = []
   for use, cols, steps in ((use_rad, st.rad_cols, rad_steps), (use_clin, st.clin_cols, clin_steps)):
       if not use:
           out += [None, 0]; continue
       pipe = Pipeline(steps()).fit(st.xtab.loc[tr, cols], y)
       X = pipe.transform(st.xtab.loc[st.pids, cols]).astype(np.float32)
       out += [dict(zip(st.pids, X)), X.shape[1]]
   return out   # rad_dict, n_rad, clin_dict, n_clin




# =============================================================== evaluation utilities
def roc_auc_safe(y, p):
   y = np.asarray(y)
   return roc_auc_score(y, p) if len(np.unique(y)) > 1 else np.nan




def compute_metrics(y, p, thr=0.5):
   y, p = np.asarray(y).astype(int), np.asarray(p, dtype=float)
   pred = (p >= thr).astype(int)
   tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
   sens = tp / (tp + fn) if (tp + fn) else np.nan
   spec = tn / (tn + fp) if (tn + fp) else np.nan
   prec = tp / (tp + fp) if (tp + fp) else np.nan
   f1 = 2 * prec * sens / (prec + sens) if (prec == prec and sens == sens and (prec + sens) > 0) else np.nan
   return {"accuracy": (tp + tn) / len(y), "balanced_acc": np.nanmean([sens, spec]), "precision": prec,
           "sensitivity": sens, "specificity": spec, "f1": f1, "roc_auc": roc_auc_safe(y, p),
           "pr_auc": average_precision_score(y, p) if len(np.unique(y)) > 1 else np.nan,
           "brier": brier_score_loss(y, p)}




def bootstrap_ci(y, p, fn, B=1000, seed=0):
   """Percentile 95% CI by resampling PATIENTS."""
   y, p = np.asarray(y), np.asarray(p)
   rng, vals = np.random.RandomState(seed), []
   for _ in range(B):
       idx = rng.randint(0, len(y), len(y))
       if len(np.unique(y[idx])) > 1:
           vals.append(fn(y[idx], p[idx]))
   return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else (np.nan, np.nan)




def summarize_oof(st, oof_dict, B=1000):
   rows = {}
   for name, df in oof_dict.items():
       m = compute_metrics(df["label"], df["prob"])
       m["roc_auc_lo"], m["roc_auc_hi"] = bootstrap_ci(df["label"], df["prob"], roc_auc_score, B)
       m["pr_auc_lo"], m["pr_auc_hi"] = bootstrap_ci(df["label"], df["prob"], average_precision_score, B)
       m["c_index"] = c_index(df["patient_id"].map(st.time), df["patient_id"].map(st.event), df["prob"])
       m["n_patients"] = len(df)
       rows[name] = m
   return pd.DataFrame(rows).T




def load_oof(names):
   """Out-of-fold patient predictions written by scripts 04-06 (missing files are skipped)."""
   out = {}
   for n in names:
       p = f"{RES_DIR}/oof_{n}.csv"
       if os.path.exists(p):
           out[n] = pd.read_csv(p, dtype={"patient_id": str})
   return out




def load_hist(names):
   out = {}
   for n in names:
       p = f"{RES_DIR}/history_{n}.csv"
       if os.path.exists(p):
           out[n] = pd.read_csv(p)
   return out




def load_oof_miss(names):
   """{model: {mode: df}} from oof_miss_<model>_<mode>.csv written by 05_train.py."""
   out = {}
   for n in names:
       d = {m: pd.read_csv(f"{RES_DIR}/oof_miss_{n}_{m}.csv", dtype={"patient_id": str})
            for m in ("ct", "rad", "clin") if os.path.exists(f"{RES_DIR}/oof_miss_{n}_{m}.csv")}
       if d:
           out[n] = d
   return out




# =============================================================== models (imported lazily: torch only where needed)
def torch_parts():
   """Build the torch models/dataset once torch is imported (keeps scripts 01-04 torch-free)."""
   import torch
   import torch.nn as nn
   import torch.nn.functional as F
   import torchvision
   from torch.utils.data import Dataset, DataLoader


   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   imnet_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
   imnet_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


   def seed_everything(s):
       random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


   def prep(x):                         # 3-view crop in [0, 1] -> ImageNet normalisation
       return (x - imnet_mean.to(x.device)) / imnet_std.to(x.device)


   def gap(f):
       return F.adaptive_avg_pool2d(f, 1).flatten(1)


   class Encoder(nn.Module):
       """ImageNet ResNet-18 trunk (3 channels = axial / coronal / sagittal) returning the layer4 map."""
       def __init__(self, pretrained=True):
           super().__init__()
           try:
               base = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
           except Exception as e:
               print("Pretrained weights unavailable (is Internet on?) - using random init:", type(e).__name__)
               base = torchvision.models.resnet18(weights=None)
           self.body = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool, base.layer1, base.layer2, base.layer3)
           self.layer4 = base.layer4


       def forward(self, x):
           return self.layer4(self.body(prep(x)))


   def mlp(n_in, d):
       return nn.Sequential(nn.Linear(n_in, d), nn.ReLU(), nn.Dropout(0.2))


   class CTNet(nn.Module):
       def __init__(self, n_rad=0, n_clin=0, pretrained=True):
           super().__init__()
           self.enc, self.drop, self.fc = Encoder(pretrained), nn.Dropout(0.3), nn.Linear(512, 2)


       def forward(self, x, rad, clin):
           return self.fc(self.drop(gap(self.enc(x))))


   class MidNet(nn.Module):
       """Intermediate fusion: CNN embedding concatenated with radiomic (+ clinical) embeddings."""
       def __init__(self, n_rad, n_clin=0, pretrained=True):
           super().__init__()
           self.enc, self.rad, self.n_clin = Encoder(pretrained), mlp(n_rad, 32), n_clin
           if n_clin:
               self.clin = mlp(n_clin, 32)
           self.head = nn.Sequential(nn.Dropout(0.4), nn.Linear(512 + 32 + (32 if n_clin else 0), 128), nn.ReLU(),
                                     nn.Dropout(0.3), nn.Linear(128, 2))


       def forward(self, x, rad, clin):
           feats = [gap(self.enc(x)), self.rad(rad)] + ([self.clin(clin)] if self.n_clin else [])
           return self.head(torch.cat(feats, 1))


   class GatedNet(nn.Module):
       """Attention-style fusion: CT, radiomic and clinical embeddings projected to d dims, a learned softmax gate
       weights the three modalities per patient, the weighted sum goes to the classifier."""
       def __init__(self, n_rad, n_clin, d=64, pretrained=True):
           super().__init__()
           self.enc = Encoder(pretrained)
           self.p_ct, self.p_rad, self.p_clin = nn.Linear(512, d), mlp(n_rad, d), mlp(n_clin, d)
           self.gate = nn.Sequential(nn.Linear(3 * d, 32), nn.ReLU(), nn.Linear(32, 3))
           self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(d, 2))
           self.last_gate = None


       def forward(self, x, rad, clin):
           E = torch.stack([self.p_ct(gap(self.enc(x))), self.p_rad(rad), self.p_clin(clin)], 1)   # (B, 3, d)
           w = torch.softmax(self.gate(E.flatten(1)), dim=1)
           self.last_gate = w.detach()
           return self.head((w.unsqueeze(-1) * E).sum(1))


   specs = {
       "ct_cnn": dict(cls=CTNet, rad=False, clin=False, md=0.0),
       "ct_rad_mid": dict(cls=MidNet, rad=True, clin=False, md=0.0),
       "ct_rad_clin_mid": dict(cls=MidNet, rad=True, clin=True, md=0.0),
       "gated": dict(cls=GatedNet, rad=True, clin=True, md=0.0),
       "gated_md": dict(cls=GatedNet, rad=True, clin=True, md=MOD_DROP),      # proposed
   }


   def build_model(name, n_rad, n_clin, pretrained=True):
       return specs[name]["cls"](n_rad, n_clin, pretrained=pretrained).to(device)


   def blank(x, rad, clin, which):
       """Missing modality = zeros (image) / training-mean (standardised tabular vector = 0)."""
       x, rad, clin = x.clone(), rad.clone(), clin.clone()
       if which == "ct":
           x.zero_()
       elif which == "rad":
           rad.zero_()
       elif which == "clin":
           clin.zero_()
       return x, rad, clin


   def modality_dropout(x, rad, clin, p):
       """Per sample, drop at most one modality: CT with prob p, radiomics with prob p, clinical with prob p."""
       r = torch.rand(x.shape[0], device=x.device)
       x, rad, clin = x.clone(), rad.clone(), clin.clone()
       x[r < p] = 0.0
       rad[(r >= p) & (r < 2 * p)] = 0.0
       clin[(r >= 2 * p) & (r < 3 * p)] = 0.0
       return x, rad, clin


   class CropDataset(Dataset):
       def __init__(self, st, pids, rad=None, clin=None, augment=False):
           self.st, self.pids, self.rad, self.clin, self.augment = st, list(pids), rad, clin, augment
           self.index = [(pi, j) for pi, pid in enumerate(self.pids) for j in range(st.crops[pid][0].shape[0])]


       def __len__(self):
           return len(self.index)


       def __getitem__(self, i):
           pi, j = self.index[i]
           pid = self.pids[pi]
           x = self.st.crops[pid][0][j]
           if self.augment:
               if random.random() < 0.5:
                   x = x[:, :, ::-1]
               if random.random() < 0.5:
                   x = x[:, ::-1, :]
               x = np.roll(x, (random.randint(-8, 8), random.randint(-8, 8)), axis=(1, 2))
           r = self.rad[pid] if self.rad is not None else np.zeros(1, np.float32)
           c = self.clin[pid] if self.clin is not None else np.zeros(1, np.float32)
           return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(r), torch.from_numpy(c), self.st.labels[pid], pi


   def make_loader(ds, train):
       return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=train, num_workers=0, drop_last=(train and len(ds) > BATCH_SIZE))


   @torch.no_grad()
   def predict(model, st, pids, rad=None, clin=None, missing=None):
       """Patient-level probability = mean over that patient's crops. missing in {None, 'ct', 'rad', 'clin'}."""
       ds = CropDataset(st, pids, rad, clin)
       model.eval()
       probs, pidx, loss_sum = [], [], 0.0
       for x, r, c, y, pi in make_loader(ds, False):
           x, r, c = x.to(device), r.to(device), c.to(device)
           if missing:
               x, r, c = blank(x, r, c, missing)
           logits = model(x, r, c)
           loss_sum += F.cross_entropy(logits, y.to(device), reduction="sum").item()
           probs.append(torch.softmax(logits, 1)[:, 1].cpu().numpy()); pidx.append(pi.numpy())
       g = pd.DataFrame({"pi": np.concatenate(pidx), "p": np.concatenate(probs)}).groupby("pi")["p"].mean()
       out = pd.DataFrame({"patient_id": [ds.pids[i] for i in g.index], "prob": g.values})
       out["label"] = out["patient_id"].map(st.labels).astype(int)
       return out, loss_sum / max(len(ds), 1)


   def train_fold(name, st, fold):
       spec = specs[name]
       seed_everything(SEED + fold)
       tr, va = split_train_val(st, fold)
       te = [p for p in st.pids if st.fold_of[p] == fold]
       rad, n_rad, clin, n_clin = fit_vectors(st, tr, spec["rad"], spec["clin"])
       model = build_model(name, n_rad, n_clin)
       cnt = np.bincount([st.labels[p] for p in tr], minlength=2)
       crit = nn.CrossEntropyLoss(weight=torch.tensor(len(tr) / (2.0 * np.maximum(cnt, 1)), dtype=torch.float32, device=device))
       opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
       sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(EPOCHS, 1))
       loader = make_loader(CropDataset(st, tr, rad, clin, augment=True), True)


       hist, best_loss, best_state = [], float("inf"), None
       for ep in range(EPOCHS):
           model.train()
           run, n = 0.0, 0
           for x, r, c, y, _ in loader:
               x, r, c, y = x.to(device), r.to(device), c.to(device), y.to(device)
               if spec["md"] > 0:
                   x, r, c = modality_dropout(x, r, c, spec["md"])
               opt.zero_grad()
               loss = crit(model(x, r, c), y)
               loss.backward()
               torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
               opt.step()
               run += loss.item() * len(y); n += len(y)
           sched.step()
           vdf, vloss = predict(model, st, va, rad, clin)
           hist.append({"epoch": ep + 1, "train_loss": run / max(n, 1), "val_loss": vloss,
                        "val_auc": roc_auc_safe(vdf["label"], vdf["prob"])})
           if vloss < best_loss:                              # model selection on validation loss
               best_loss, best_state = vloss, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
       model.load_state_dict(best_state)
       test_df = predict(model, st, te, rad, clin)[0].assign(fold=fold)
       modes = ["ct"] + (["rad"] if spec["rad"] else []) + (["clin"] if spec["clin"] else [])
       miss = {m: predict(model, st, te, rad, clin, missing=m)[0].assign(fold=fold) for m in modes} if len(modes) > 1 else {}
       if fold == 0 and name in SAVE_WEIGHTS:
           torch.save(best_state, f"{WEIGHT_DIR}/{name}_fold0.pt")
       return test_df, pd.DataFrame(hist).assign(model=name, fold=fold), miss, (len(tr), len(va), len(te))


   def load_fold0(name, st):
       """Fold-0 weights + the tabular vectors fitted on the same training patients as during training."""
       p = f"{WEIGHT_DIR}/{name}_fold0.pt"
       if not os.path.exists(p):
           return None, None, None
       spec = specs[name]
       rad, n_rad, clin, n_clin = fit_vectors(st, split_train_val(st, 0)[0], spec["rad"], spec["clin"])
       model = build_model(name, n_rad, n_clin, pretrained=False)
       model.load_state_dict(torch.load(p, map_location=device)); model.eval()
       return model, rad, clin


   return dict(torch=torch, F=F, device=device, specs=specs, build_model=build_model, predict=predict,
               train_fold=train_fold, load_fold0=load_fold0)
