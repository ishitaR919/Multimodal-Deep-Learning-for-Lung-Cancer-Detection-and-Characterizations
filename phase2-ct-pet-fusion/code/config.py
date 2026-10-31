"""Shared paths and settings for all Phase 2 scripts.


Defaults match Kaggle. To run elsewhere, set the environment variables, e.g.
   INPUT_DIR=/data WORK_DIR=./work python 01_build_cohort.py
"""
import os


INPUT_DIR = os.environ.get("INPUT_DIR", "/kaggle/input")         # searched for the LUNG-PET-CT-Dx dataset
WORK_DIR = os.environ.get("WORK_DIR", "/kaggle/working")
ROOT_CANDIDATE = os.environ.get("DATA_ROOT", f"{INPUT_DIR}/datasets/rangan2510/lung-pet-ct-dx")


# run mode
QUICK_TEST = False           # True: ~24 patients, 2 folds, 2 epochs (pipeline check). False = final run.
QUICK_N_PATIENTS = 24        # cohort size in quick mode (stratified random sample)
TASK = "adeno_vs_rest"       # or "adeno_vs_squamous"
SEED = 42


# training
N_FOLDS = 2 if QUICK_TEST else 5
EPOCHS = 2 if QUICK_TEST else 25
BATCH_SIZE = 32
LR = 3e-4


# preprocessing
SLICES_PER_PATIENT = 7       # annotated slices sampled per patient (patient prediction = mean over slices)
CROP_FOV_MM = 160.0          # physical side length of the square crop around the tumor box centre
CROP_PX = 128                # crop resolution (CT and PET resampled to the SAME grid)
CT_WIN = (-1000.0, 400.0)    # HU window -> [0, 1]
PET_SUV_MAX = 15.0           # SUV clipped to [0, PET_SUV_MAX] -> [0, 1]
RECOMPUTE_ANNOTATIONS = False  # True: ignore cached annotation check
MIN_MATCHED_SLICES = 3       # annotated slices that must be found in the CT of the PET study (crops use up to 7)
ANN_THREADS = 32             # header reads are I/O bound (network disk) -> threads overlap the waits
PREP_WORKERS = os.cpu_count() or 1
SAVE_WEIGHTS = ["early", "intermediate", "cross_attn_md"]   # fold-0 weights kept for explainability


SUBTYPE_MAP = {"A": "Adenocarcinoma", "B": "SmallCell", "E": "LargeCell", "G": "Squamous"}
CLIN_COLS = ["sex_male", "age", "weight_kg", "smoking", "t_num", "n_num", "m1"]   # 'grade' dropped: ~60% missing
RAD_COLS = ["suv_max", "suv_mean", "suv_std", "hu_mean", "hu_std", "box_w_mm", "box_h_mm", "z_extent_mm"]


# models: kind, CT/PET channel (uni), clinical branch, modality-dropout prob, uses both modalities
MODEL_SPECS = {
   "ct_only": dict(kind="uni", channel=0, clin=False, md=0.0, both=False),
   "pet_only": dict(kind="uni", channel=1, clin=False, md=0.0, both=False),
   "early": dict(kind="early", clin=False, md=0.0, both=True),
   "intermediate": dict(kind="mid", clin=False, md=0.0, both=True),
   "intermediate_clin": dict(kind="mid", clin=True, md=0.0, both=True),
   "cross_attn": dict(kind="xattn", clin=False, md=0.0, both=True),
   "cross_attn_md": dict(kind="xattn", clin=False, md=0.25, both=True),       # proposed: + modality dropout
   "cross_attn_md_clin": dict(kind="xattn", clin=True, md=0.25, both=True),   # proposed + clinical
}
TAB_MODELS = ["tab_clinical_lr", "tab_radiomics_lr", "tab_clin_radiomics_lr", "tab_clin_radiomics_rf"]


# paths
RES_DIR = f"{WORK_DIR}/phase2_results"
VOL_DIR = f"{RES_DIR}/volumes"
CROP_DIR = f"{RES_DIR}/crops"
WEIGHT_DIR = f"{RES_DIR}/weights"
FIG_DIR = f"{WORK_DIR}/figures/phase2"
_sfx = "_quick" if QUICK_TEST else ""          # quick runs check only a sample -> never reuse their cache for the full run
V2_CSV = f"{RES_DIR}/paired_patients_v2{_sfx}.csv"
BOXES_CSV = f"{RES_DIR}/annotation_boxes{_sfx}.csv"
SUMM_CSV = f"{RES_DIR}/annotation_patient_summary{_sfx}.csv"
LOG_CSV = f"{RES_DIR}/preprocess_log.csv"
for _d in (RES_DIR, VOL_DIR, CROP_DIR, WEIGHT_DIR, FIG_DIR):
   os.makedirs(_d, exist_ok=True)