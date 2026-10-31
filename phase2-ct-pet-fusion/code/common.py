"""Data, preprocessing, metric and state-loading code shared by the Phase 2 scripts (no torch here).


State is never kept in module globals: every function takes what it needs as arguments, and
`load_state()` rebuilds the cohort, crops, master table and folds from the files written by earlier steps.
"""
import glob
import json
import os
import re
import time
import unicodedata
import xml.etree.ElementTree as ET
from types import SimpleNamespace


import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, brier_score_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


from config import (INPUT_DIR, ROOT_CANDIDATE, QUICK_TEST, QUICK_N_PATIENTS, TASK, SEED, N_FOLDS,
                   SLICES_PER_PATIENT, CROP_FOV_MM, CROP_PX, CT_WIN, PET_SUV_MAX, MIN_MATCHED_SLICES,
                   SUBTYPE_MAP, CLIN_COLS, VOL_DIR, CROP_DIR, RES_DIR, V2_CSV, LOG_CSV)




# ==================== dataset tables and CT/PET pairing ====================
def find_root():
   if os.path.exists(os.path.join(ROOT_CANDIDATE, "metadata.csv")):
       return ROOT_CANDIDATE
   for r, _, files in os.walk(INPUT_DIR):
       if "metadata.csv" in files and "statistics-clinical-20201221.xlsx" in files:
           return r
   raise FileNotFoundError("LUNG-PET-CT-Dx dataset not found under INPUT_DIR - add it as notebook input.")


def resolve_series_dir(file_location, img_root):
   p = re.sub(r"^\./", "", str(file_location).replace("\\", "/"))
   parts = [x for x in p.split("/") if x]
   if parts and parts[0] == "Lung-PET-CT-Dx":
       parts = parts[1:]
   return os.path.join(img_root, *parts)


def load_clinical(root):
   clin = pd.read_excel(os.path.join(root, "statistics-clinical-20201221.xlsx"))
   clin.columns = [unicodedata.normalize("NFKC", str(c)).strip() for c in clin.columns]  # full-width 'M' in 'M-Stage'
   clin = clin.rename(columns={"NewPatientID": "patient_id", "Sex": "sex", "Age": "age", "weight (kg)": "weight_kg",
                               "T-Stage": "t_stage", "N-Stage": "n_stage", "M-Stage": "m_stage",
                               "Histopathological grading": "grade", "Smoking History": "smoking"})
   clin["patient_id"] = clin["patient_id"].astype(str).str.strip()
   return clin.drop(columns=[c for c in ["No."] if c in clin.columns])


def build_series_table(root):
   meta = pd.read_csv(os.path.join(root, "metadata.csv"))
   img_root = os.path.join(root, "Lung-PET-CT-Dx", "Lung-PET-CT-Dx")
   s = meta.rename(columns={"Series UID": "series_uid", "Study UID": "study_uid", "Study Date": "study_date",
                            "Series Description": "series_desc", "Modality": "modality",
                            "Number of Images": "n_images"}).copy()
   s["patient_id"] = s["Subject ID"].str.replace("Lung_Dx-", "", regex=False)
   s["series_dir"] = s["File Location"].apply(lambda x: resolve_series_dir(x, img_root))
   s["dir_exists"] = s["series_dir"].apply(os.path.isdir)
   s["study_date_dt"] = pd.to_datetime(s["study_date"], format="%m-%d-%Y", errors="coerce")
   return s[["patient_id", "series_uid", "study_uid", "study_date", "study_date_dt", "series_desc",
             "modality", "n_images", "series_dir", "dir_exists"]]


def pair_patients(s):
   """Patients with CT and PET in the SAME study; CT/PET series with most images (provisional rule)."""
   rows = []
   for pid, g in s.groupby("patient_id"):
       for study_uid, sg in g.sort_values("study_date_dt").groupby("study_uid", sort=False):
           ct, pt = sg[sg["modality"] == "CT"], sg[sg["modality"] == "PT"]
           if len(ct) and len(pt):
               c = ct.sort_values("n_images", ascending=False).iloc[0]
               p = pt.sort_values("n_images", ascending=False).iloc[0]
               rows.append({"patient_id": pid, "study_uid": study_uid, "study_date": c["study_date"],
                            "n_ct_series_in_study": len(ct), "n_pt_series_in_study": len(pt),
                            "ct_series_uid": c["series_uid"], "ct_desc": c["series_desc"], "ct_n_images": c["n_images"],
                            "ct_dir": c["series_dir"], "pt_series_uid": p["series_uid"], "pt_desc": p["series_desc"],
                            "pt_n_images": p["n_images"], "pt_dir": p["series_dir"]})
               break
   return pd.DataFrame(rows)


def add_subtype(df):
   df = df.copy()
   df["subtype_letter"] = df["patient_id"].str[0]
   df["subtype"] = df["subtype_letter"].map(SUBTYPE_MAP)
   return df


def make_labels(df, task=TASK):
   df = df.copy()
   if task == "adeno_vs_rest":
       df["label"] = (df["subtype"] != "Adenocarcinoma").astype(int)
   elif task == "adeno_vs_squamous":
       df = df[df["subtype"].isin(["Adenocarcinoma", "Squamous"])].copy()
       df["label"] = (df["subtype"] == "Squamous").astype(int)
   else:
       raise ValueError(task)
   return df


# ==================== tumour annotations (XML boxes) and the annotated CT series ====================




def parse_annotation_xml(path):
   out = []
   for obj in ET.parse(path).getroot().iter("object"):
       bb = obj.find("bndbox")
       if bb is None:
           continue
       try:
           xmin, ymin, xmax, ymax = [int(float(bb.findtext(k))) for k in ("xmin", "ymin", "xmax", "ymax")]
       except (TypeError, ValueError):
           continue
       out.append(((obj.findtext("name") or "").strip().lower(), xmin, ymin, xmax, ymax))
   return out


def load_patient_boxes(ann_root, pid):
   rows = []
   for xml_path in sorted(glob.glob(os.path.join(ann_root, pid, "**", "*.xml"), recursive=True)):
       sop_uid = os.path.splitext(os.path.basename(xml_path))[0]   # XML file name = CT slice SOP Instance UID
       for name, x0, y0, x1, y1 in parse_annotation_xml(xml_path):
           rows.append({"patient_id": pid, "sop_uid": sop_uid, "class_name": name,
                        "xmin": x0, "ymin": y0, "xmax": x1, "ymax": y1})
   return pd.DataFrame(rows)


def pick_annotated_series(found):
   if not found:
       return None, 0
   counts = pd.Series([v["series_uid"] for v in found.values()]).value_counts()
   return counts.index[0], int(counts.iloc[0])


def roi_summary(matched):
   if matched.empty:
       return {}
   z = matched["ipp_z"]
   return {"n_annotated_slices": int(matched["sop_uid"].nunique()),
           "max_boxes_per_slice": int(matched.groupby("sop_uid").size().max()),
           "z_min_mm": float(z.min()), "z_max_mm": float(z.max()), "z_extent_mm": float(z.max() - z.min()),
           "mean_box_w_px": float((matched["xmax"] - matched["xmin"]).mean()),
           "mean_box_h_px": float((matched["ymax"] - matched["ymin"]).mean())}


def locate_slices(sop_uids, ct_series_rows, preferred_series_uid):
   """Header-only DICOM read: which CT series holds each annotated SOP UID (+ slice position)."""
   import pydicom
   remaining, found = set(sop_uids), {}
   for r in sorted(ct_series_rows, key=lambda r: r["series_uid"] != preferred_series_uid):
       for f in sorted(glob.glob(os.path.join(r["series_dir"], "*.dcm"))):
           if not remaining:
               break
           try:
               d = pydicom.dcmread(f, stop_before_pixels=True,
                                   specific_tags=["SOPInstanceUID", "ImagePositionPatient", "InstanceNumber"])
           except Exception:
               continue
           sop = str(d.SOPInstanceUID)
           if sop in remaining:
               ipp = [float(x) for x in getattr(d, "ImagePositionPatient", [np.nan] * 3)]
               found[sop] = {"series_uid": r["series_uid"], "ipp_x": ipp[0], "ipp_y": ipp[1], "ipp_z": ipp[2],
                             "instance_number": int(getattr(d, "InstanceNumber", -1))}
               remaining.discard(sop)
       if len(remaining) <= 0.1 * len(sop_uids):
           break
   return found


def check_patient(row, ann_root, ct_series, series_by_uid):
   """Find which CT series holds this patient's annotated slices (header-only reads)."""
   pid = row["patient_id"]
   boxes = load_patient_boxes(ann_root, pid)
   summ = {"patient_id": pid, "selected_ct_series_uid": row["ct_series_uid"], "n_xml": 0,
           "n_boxes": len(boxes), "match_frac": 0.0, "annot_series_uid": None}
   if not boxes.empty:
       sops = boxes["sop_uid"].unique().tolist()
       summ["n_xml"] = len(sops)
       found = locate_slices(sops, ct_series[ct_series["patient_id"] == pid].to_dict("records"), row["ct_series_uid"])
       annot_uid, hits = pick_annotated_series(found)
       summ["match_frac"], summ["annot_series_uid"], summ["n_matched"] = round(hits / len(sops), 3), annot_uid, hits
       info = (pd.DataFrame.from_dict(found, orient="index").rename_axis("sop_uid").reset_index()
               if found else pd.DataFrame(columns=["sop_uid"]))
       boxes = boxes.merge(info, on="sop_uid", how="left")
       if annot_uid is not None:
           summ.update(roi_summary(boxes[boxes["series_uid"] == annot_uid]))
           summ["annot_ct_dir"] = series_by_uid.loc[annot_uid, "series_dir"]
           summ["annot_study_uid"] = series_by_uid.loc[annot_uid, "study_uid"]
           summ["annot_n_images"] = int(series_by_uid.loc[annot_uid, "n_images"])
       summ["class_names"] = ",".join(sorted(set(boxes["class_name"])))
   return boxes, summ




def finalize_annotation_check(paired, summ_df):
   """Fusion rule + CT switch: usable = enough annotated slices in a CT series of the same study as the PET.
   Removes volumes / crops built earlier from a different CT series. Returns (v2, summ_df)."""
   if "n_matched" not in summ_df:                 # caches written before n_matched was stored
       summ_df["n_matched"] = (summ_df["match_frac"] * summ_df["n_xml"]).round()
   summ_df["annot_in_pet_study"] = summ_df["annot_in_pet_study"].fillna(False).astype(bool)
   summ_df["fusion_ok"] = (summ_df["n_matched"].fillna(0) >= MIN_MATCHED_SLICES) & summ_df["annot_in_pet_study"]


   v2 = paired.merge(summ_df[["patient_id", "annot_series_uid", "annot_ct_dir", "annot_n_images", "match_frac", "n_matched",
                              "matches_selected_ct", "annot_in_pet_study", "fusion_ok"]], on="patient_id", how="left")
   v2["fusion_ok"] = v2["fusion_ok"].fillna(False).astype(bool)
   v2["ct_dir_orig"] = v2["ct_dir"]
   switch = v2["fusion_ok"] & ~v2["matches_selected_ct"].fillna(True).astype(bool)
   v2.loc[switch, ["ct_series_uid", "ct_dir", "ct_n_images"]] = v2.loc[switch, ["annot_series_uid", "annot_ct_dir", "annot_n_images"]].values
   print(f"CT series switched to the annotated series for {int(switch.sum())} patients.")
   if os.path.exists(V2_CSV):                     # volumes / crops built from a different CT series are stale
       _prev = pd.read_csv(V2_CSV).set_index("patient_id")["ct_series_uid"]
       stale = [p for p, uid in zip(v2["patient_id"], v2["ct_series_uid"]) if p in _prev.index and _prev[p] != uid]
       for pid in stale:
           for f in (f"{VOL_DIR}/{pid}_ct.nii.gz", f"{CROP_DIR}/{pid}.npz"):
               if os.path.exists(f):
                   os.remove(f)
       if stale and os.path.exists(LOG_CSV):
           _log = pd.read_csv(LOG_CSV); _log[~_log["patient_id"].isin(stale)].to_csv(LOG_CSV, index=False)
   return v2, summ_df




def define_cohort(v2):
   """Usable patients + labels (+ stratified sample in QUICK_TEST mode). Deterministic (seeded)."""
   cohort = make_labels(v2[v2["fusion_ok"]].reset_index(drop=True))
   if QUICK_TEST and len(cohort) > QUICK_N_PATIENTS:
       parts = [g.sample(min(len(g), max(5, round(QUICK_N_PATIENTS * len(g) / len(cohort)))), random_state=SEED)
                for _, g in cohort.groupby("label")]
       cohort = pd.concat(parts).sort_values("patient_id").reset_index(drop=True)
   return cohort


# ==================== DICOM -> CT (HU) and PET (SUV) volumes ====================




def parse_dicom_time(value):
   s = str(value).strip()
   whole, frac = (s.split(".", 1)[0], float("0." + s.split(".", 1)[1])) if "." in s else (s, 0.0)
   whole = whole.ljust(6, "0")
   return int(whole[0:2]) * 3600 + int(whole[2:4]) * 60 + int(whole[4:6]) + frac


def suv_bw_factor(weight_kg, total_dose_bq, half_life_s, injection_time_s, scan_time_s, decay_to_scan=True):
   """Multiply activity concentration (Bq/mL) by this factor to obtain body-weight SUV."""
   if decay_to_scan:
       dt = scan_time_s - injection_time_s
       if dt < 0:
           dt += 86400.0
       dose = total_dose_bq * 2.0 ** (-dt / half_life_s)
   else:
       dose = total_dose_bq
   return (weight_kg * 1000.0) / dose


def sort_and_geometry(positions, iop, pixel_spacing):
   iop = np.asarray(iop, dtype=float)
   row_dir, col_dir = iop[:3], iop[3:]
   normal = np.cross(row_dir, col_dir)
   pos = np.asarray(positions, dtype=float)
   proj = pos @ normal
   order = np.argsort(proj, kind="stable")
   proj_sorted = proj[order]
   keep = np.concatenate([[True], np.diff(proj_sorted) > 1e-3])      # drop duplicated slice positions
   order, proj_sorted = order[keep], proj_sorted[keep]
   dz = np.diff(proj_sorted)
   gap = float(np.median(dz)) if len(dz) else 1.0
   dev = float(np.max(np.abs(dz - gap))) if len(dz) else 0.0
   spacing = (float(pixel_spacing[1]), float(pixel_spacing[0]), gap)  # DICOM PixelSpacing = [row(y), col(x)]
   direction = (row_dir[0], col_dir[0], normal[0], row_dir[1], col_dir[1], normal[1], row_dir[2], col_dir[2], normal[2])
   return order, spacing, tuple(float(x) for x in pos[order[0]]), tuple(float(x) for x in direction), dev


def read_series(series_dir):
   import pydicom
   files = sorted(glob.glob(os.path.join(series_dir, "*.dcm"))) or \
       sorted(f for f in glob.glob(os.path.join(series_dir, "*")) if os.path.isfile(f))
   dsets = []
   for f in files:
       d = pydicom.dcmread(f, force=True)
       if hasattr(d, "ImagePositionPatient") and hasattr(d, "PixelData"):
           dsets.append(d)
   if not dsets:
       raise ValueError(f"no usable DICOM slices in {series_dir}")
   order, spacing, origin, direction, dev = sort_and_geometry([d.ImagePositionPatient for d in dsets],
                                                              dsets[0].ImageOrientationPatient, dsets[0].PixelSpacing)
   dsets = [dsets[i] for i in order]
   vol = np.stack([d.pixel_array.astype(np.float32) * float(getattr(d, "RescaleSlope", 1.0)) +
                   float(getattr(d, "RescaleIntercept", 0.0)) for d in dsets], axis=0)   # per-slice rescale
   return vol, dsets, spacing, origin, direction, dev


def to_sitk(vol, spacing, origin, direction, dtype):
   import SimpleITK as sitk
   img = sitk.GetImageFromArray(vol.astype(dtype))
   img.SetSpacing(spacing); img.SetOrigin(origin); img.SetDirection(direction)
   return img


def pet_to_suv(vol, dsets, fallback_weight_kg=None):
   d0 = dsets[0]
   units = str(getattr(d0, "Units", "")).strip().upper()
   info = {"pet_units": units}
   if units == "GML":
       info["suv_method"] = "units_gml"; return vol, info
   if units == "CNTS":                                    # Philips private SUV scale factor
       scale = d0.get((0x7053, 0x1000))
       if scale is None:
           raise ValueError("PET units CNTS without Philips SUV scale factor")
       info["suv_method"] = "philips_private"; return vol * float(scale.value), info
   if units != "BQML":
       raise ValueError(f"unsupported PET units '{units}'")
   weight = float(getattr(d0, "PatientWeight", 0) or 0)
   info["weight_source"] = "dicom"
   if not weight:
       if fallback_weight_kg is None or pd.isna(fallback_weight_kg):
           raise ValueError("no patient weight in DICOM or clinical sheet")
       weight, info["weight_source"] = float(fallback_weight_kg), "clinical_sheet"
   rp = d0.RadiopharmaceuticalInformationSequence[0]
   dose, half_life = float(rp.RadionuclideTotalDose), float(rp.RadionuclideHalfLife)
   inj_t = parse_dicom_time(rp.RadiopharmaceuticalStartTime)
   times = [parse_dicom_time(d.AcquisitionTime) for d in dsets if hasattr(d, "AcquisitionTime")]
   scan_t = min(times) if times else parse_dicom_time(d0.SeriesTime)
   decay_corr = str(getattr(d0, "DecayCorrection", "START")).strip().upper()
   factor = suv_bw_factor(weight, dose, half_life, inj_t, scan_t, decay_to_scan=(decay_corr != "ADMIN"))
   info.update({"suv_method": "bqml_bodyweight", "weight_kg": weight, "dose_mbq": dose / 1e6,
                "uptake_min": ((scan_t - inj_t) % 86400) / 60.0, "decay_correction": decay_corr})
   return vol * factor, info


def process_patient(row):
   import SimpleITK as sitk
   t0, pid = time.time(), row["patient_id"]
   ct, ct_ds, ct_sp, ct_org, ct_dir, ct_dev = read_series(row["ct_dir"])
   pet, pet_ds, pet_sp, pet_org, pet_dir, pet_dev = read_series(row["pt_dir"])
   suv, info = pet_to_suv(pet, pet_ds, fallback_weight_kg=row.get("weight_kg"))
   ct_img = to_sitk(np.clip(ct, -1024, 3071), ct_sp, ct_org, ct_dir, np.int16)
   pet_img = to_sitk(suv, pet_sp, pet_org, pet_dir, np.float32)
   res = sitk.Resample(pet_img, ct_img, sitk.Transform(), sitk.sitkLinear, -1.0, sitk.sitkFloat32)
   coverage = float((sitk.GetArrayFromImage(res) > -0.5).mean())          # share of the CT grid inside the PET FOV
   sitk.WriteImage(ct_img, f"{VOL_DIR}/{pid}_ct.nii.gz")
   sitk.WriteImage(pet_img, f"{VOL_DIR}/{pid}_pet_suv.nii.gz")
   rec = {"patient_id": pid, "status": "ok", "error": ""}
   rec.update(info)
   rec.update({"ct_shape_zyx": str(ct.shape), "ct_spacing_xyz": str(tuple(round(x, 3) for x in ct_sp)),
               "pet_shape_zyx": str(pet.shape), "pet_spacing_xyz": str(tuple(round(x, 3) for x in pet_sp)),
               "same_frame_of_reference": getattr(ct_ds[0], "FrameOfReferenceUID", "ct") == getattr(pet_ds[0], "FrameOfReferenceUID", "pet"),
               "suv_max": float(suv.max()), "suv_p999": float(np.percentile(suv, 99.9)),
               "ct_pet_coverage": round(coverage, 3), "seconds": round(time.time() - t0, 1)})
   return rec


def _safe_process_patient(row):
   """Worker wrapper: one failed patient never stops the run."""
   try:
       return process_patient(row)
   except Exception as e:
       return {"patient_id": row["patient_id"], "status": "failed", "error": f"{type(e).__name__}: {e}"}


# ==================== aligned CT/PET tumour crops ====================




def make_ref(ct_img, center_phys, fov, npx):
   """Reference grid: npx x npx x 1, in-plane spacing fov/npx, centred on `center_phys`, CT orientation."""
   import SimpleITK as sitk
   D = np.array(ct_img.GetDirection()).reshape(3, 3)
   sp = (fov / npx, fov / npx, float(ct_img.GetSpacing()[2]))
   half = (npx - 1) / 2.0
   origin = np.array(center_phys) - D[:, 0] * sp[0] * half - D[:, 1] * sp[1] * half
   ref = sitk.Image(npx, npx, 1, sitk.sitkFloat32)
   ref.SetSpacing(sp); ref.SetOrigin(tuple(float(v) for v in origin)); ref.SetDirection(ct_img.GetDirection())
   return ref


def select_boxes(boxes_p, n_slices):
   """One (largest) box per annotated slice; pick n slices evenly over the central 80% of the tumor extent."""
   b = boxes_p.dropna(subset=["ipp_z"]).copy()
   b["area"] = (b.xmax - b.xmin) * (b.ymax - b.ymin)
   b = b.sort_values("area", ascending=False).drop_duplicates("sop_uid").sort_values("ipp_z").reset_index(drop=True)
   m = len(b)
   if m <= n_slices:
       return b
   lo, hi = int(0.1 * (m - 1)), int(round(0.9 * (m - 1)))
   return b.iloc[np.unique(np.round(np.linspace(lo, hi, n_slices)).astype(int))]


def extract_patient_crops(pid, boxes_p):
   import SimpleITK as sitk
   ct = sitk.ReadImage(f"{VOL_DIR}/{pid}_ct.nii.gz")
   pet = sitk.ReadImage(f"{VOL_DIR}/{pid}_pet_suv.nii.gz")
   sx, sy = ct.GetSpacing()[0], ct.GetSpacing()[1]
   ppm = CROP_PX / CROP_FOV_MM
   yy, xx = np.mgrid[0:CROP_PX, 0:CROP_PX]
   c = (CROP_PX - 1) / 2.0
   cts, pets, feats = [], [], []
   for r in select_boxes(boxes_p, SLICES_PER_PATIENT).itertuples():
       cont = ct.TransformPhysicalPointToContinuousIndex((float(r.ipp_x), float(r.ipp_y), float(r.ipp_z)))
       k = int(np.clip(round(cont[2]), 0, ct.GetSize()[2] - 1))
       center = ct.TransformContinuousIndexToPhysicalPoint(((r.xmin + r.xmax) / 2.0, (r.ymin + r.ymax) / 2.0, float(k)))
       ref = make_ref(ct, center, CROP_FOV_MM, CROP_PX)
       ct_s = sitk.GetArrayFromImage(sitk.Resample(ct, ref, sitk.Transform(), sitk.sitkLinear, -1024.0, sitk.sitkFloat32))[0]
       pet_s = sitk.GetArrayFromImage(sitk.Resample(pet, ref, sitk.Transform(), sitk.sitkLinear, 0.0, sitk.sitkFloat32))[0]
       hw, hh = max(1.0, 0.5 * (r.xmax - r.xmin) * sx * ppm), max(1.0, 0.5 * (r.ymax - r.ymin) * sy * ppm)
       mask = (np.abs(xx - c) <= hw) & (np.abs(yy - c) <= hh)               # tumor box in crop pixels
       feats.append({"suv_max": pet_s[mask].max(), "suv_mean": pet_s[mask].mean(), "suv_std": pet_s[mask].std(),
                     "hu_mean": ct_s[mask].mean(), "hu_std": ct_s[mask].std(),
                     "box_w_mm": (r.xmax - r.xmin) * sx, "box_h_mm": (r.ymax - r.ymin) * sy})
       cts.append(np.clip((ct_s - CT_WIN[0]) / (CT_WIN[1] - CT_WIN[0]), 0, 1).astype(np.float16))
       pets.append(np.clip(pet_s / PET_SUV_MAX, 0, 1).astype(np.float16))
   f = pd.DataFrame(feats)
   patient_feat = {"suv_max": f.suv_max.max(), "suv_mean": f.suv_mean.mean(), "suv_std": f.suv_std.mean(),
                   "hu_mean": f.hu_mean.mean(), "hu_std": f.hu_std.mean(),
                   "box_w_mm": f.box_w_mm.max(), "box_h_mm": f.box_h_mm.max(),
                   "z_extent_mm": float(boxes_p["ipp_z"].max() - boxes_p["ipp_z"].min()),
                   "n_annotated_slices": int(boxes_p["sop_uid"].nunique())}
   np.savez_compressed(f"{CROP_DIR}/{pid}.npz", ct=np.stack(cts), pet=np.stack(pets),
                       feat_json=np.array(json.dumps({k: float(v) for k, v in patient_feat.items()})))


# ==================== rebuilding state from disk ====================




def _stage_num(v):
   """'2b' -> 2.2, '1c' -> 1.3, '3' -> 3.0, 'x'/NaN -> NaN"""
   m = re.match(r"(\d)([abc]?)", str(v).strip().lower())
   return np.nan if not m else int(m.group(1)) + {"": 0.0, "a": 0.1, "b": 0.2, "c": 0.3}[m.group(2)]


def build_master(cohort_df, feats):
   m = cohort_df[["patient_id", "subtype", "label", "sex", "age", "weight_kg", "smoking",
                  "t_stage", "n_stage", "m_stage"]].copy()
   m["sex_male"] = np.where(m["sex"].isna(), np.nan, (m["sex"].astype(str).str.upper().str[0] == "M").astype(float))
   m["t_num"], m["n_num"] = m["t_stage"].apply(_stage_num), m["n_stage"].apply(_stage_num)
   m["m1"] = m["m_stage"].apply(lambda v: np.nan if pd.isna(v) else float(str(v).strip().startswith("1")))
   m["smoking"] = pd.to_numeric(m["smoking"], errors="coerce")
   fe_ = pd.DataFrame.from_dict(feats, orient="index").rename_axis("patient_id").reset_index()
   return m.merge(fe_, on="patient_id", how="left")




def load_cohort(stage="crops"):
   """Cohort as the notebook has it after each step:
   'labels' (after the annotation check), 'volumes' (DICOM conversion ok), 'crops' (crop file exists)."""
   if not os.path.exists(V2_CSV):
       raise FileNotFoundError(f"{V2_CSV} missing - run 01_build_cohort.py first.")
   cohort = define_cohort(pd.read_csv(V2_CSV))
   if stage in ("volumes", "crops"):
       if not os.path.exists(LOG_CSV):
           raise FileNotFoundError(f"{LOG_CSV} missing - run 02_dicom_to_volumes.py first.")
       log = pd.read_csv(LOG_CSV)
       cohort = cohort[cohort["patient_id"].isin(log.loc[log["status"] == "ok", "patient_id"])].reset_index(drop=True)
   if stage == "crops":
       cohort = cohort[[os.path.exists(f"{CROP_DIR}/{p}.npz") for p in cohort["patient_id"]]].reset_index(drop=True)
   return cohort




def load_crops(pids):
   """CROPS[pid] = (ct, pet) float32 arrays (n_slices, H, W); FEATS[pid] = tumour ROI features."""
   crops, feats = {}, {}
   for pid in pids:
       pth = f"{CROP_DIR}/{pid}.npz"
       if os.path.exists(pth):
           z = np.load(pth)
           crops[pid] = (z["ct"].astype(np.float32), z["pet"].astype(np.float32))
           feats[pid] = json.loads(str(z["feat_json"]))
   return crops, feats




def make_folds(pids, labels):
   """Patient-level stratified K-fold: FOLD_OF[pid] = test fold index."""
   skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
   fold_of = {}
   for f, (_, te) in enumerate(skf.split(pids, [labels[p] for p in pids])):
       for i in te:
           fold_of[pids[i]] = f
   return fold_of




def load_state(with_crops=True):
   """Everything the modelling steps need, rebuilt from disk (same order and seeds as the notebook)."""
   cohort = load_cohort("crops")
   crops, feats = load_crops(cohort["patient_id"])
   cohort = cohort[cohort["patient_id"].isin(crops)].reset_index(drop=True)
   master = build_master(cohort, feats).sort_values("patient_id").reset_index(drop=True)
   pids = master["patient_id"].tolist()
   labels = dict(zip(master["patient_id"], master["label"].astype(int)))
   return SimpleNamespace(cohort=cohort, crops=crops if with_crops else None, feats=feats, master=master, pids=pids,
                          labels=labels, fold_of=make_folds(pids, labels),
                          clin_raw=master.set_index("patient_id")[CLIN_COLS])




def load_oof(names):
   """Out-of-fold patient predictions saved by earlier steps (missing files are skipped)."""
   return {n: pd.read_csv(f"{RES_DIR}/oof_{n}.csv") for n in names if os.path.exists(f"{RES_DIR}/oof_{n}.csv")}


def make_clin_dict(clin_raw, train_pids, all_pids):
   """Impute + standardise clinical features using ONLY the training patients of the fold."""
   imp = SimpleImputer(strategy="median", keep_empty_features=True)
   sc = StandardScaler()
   sc.fit(imp.fit_transform(clin_raw.loc[train_pids]))
   X = sc.transform(imp.transform(clin_raw.loc[all_pids]))
   return {p: X[i].astype(np.float32) for i, p in enumerate(all_pids)}, len(CLIN_COLS)


# ==================== evaluation utilities ====================




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
   """Percentile 95% CI by resampling PATIENTS (rows are patient-level predictions)."""
   y, p = np.asarray(y), np.asarray(p)
   rng, vals = np.random.RandomState(seed), []
   for _ in range(B):
       idx = rng.randint(0, len(y), len(y))
       if len(np.unique(y[idx])) > 1:
           vals.append(fn(y[idx], p[idx]))
   return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else (np.nan, np.nan)


def summarize_oof(oof_dict, B=1000):
   rows = {}
   for name, df in oof_dict.items():
       m = compute_metrics(df["label"], df["prob"])
       m["roc_auc_lo"], m["roc_auc_hi"] = bootstrap_ci(df["label"], df["prob"], roc_auc_score, B)
       m["pr_auc_lo"], m["pr_auc_hi"] = bootstrap_ci(df["label"], df["prob"], average_precision_score, B)
       m["n_patients"] = len(df)
       rows[name] = m
   return pd.DataFrame(rows).T


def per_fold_auc(df):
   return df.groupby("fold").apply(lambda g: roc_auc_safe(g["label"], g["prob"]))