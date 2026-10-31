"""Shared paths and settings for all Phase 3 scripts.


Defaults match Kaggle. To run elsewhere, set the environment variables, e.g.
   INPUT_DIR=/data WORK_DIR=./work IMAGE_ROOT=/data/nsclc CLINICAL_FILE=/data/lung1.csv python 01_build_cohort.py
"""
import os


INPUT_DIR = os.environ.get("INPUT_DIR", "/kaggle/input")
WORK_DIR = os.environ.get("WORK_DIR", "/kaggle/working")
# clinical CSV/XLSX with survival time + dead status ("" = search INPUT_DIR)
CLINICAL_FILE = os.environ.get(
   "CLINICAL_FILE",
   "/kaggle/input/datasets/blackhatcoder21/lung1-clinical-csv/NSCLC-Radiomics-Lung1.clinical-version3-Oct-2019.csv")
# CT + RTSTRUCT DICOM (falls back to INPUT_DIR if the folder does not exist)
IMAGE_ROOT = os.environ.get("IMAGE_ROOT", "/kaggle/input/datasets/umutkrdrms/nsclc-radiomics")


# pyradiomics if installed, else the built-in "lite" feature set. Decided once here so every script
# uses the same feature set and the same crop cache folder (override with FEATURE_SET=lite|pyradiomics).
import importlib.util  # noqa: E402
FEATURE_SET = os.environ.get("FEATURE_SET", "pyradiomics" if importlib.util.find_spec("radiomics") else "lite")
HAVE_PYRADIOMICS = FEATURE_SET == "pyradiomics"


QUICK_TEST = os.environ.get("QUICK_TEST", "0") == "1"   # 1: 40 patients, 2 folds, 2 epochs (pipeline check only)
QUICK_N_PATIENTS = 40
SEED = 42
N_FOLDS = 2 if QUICK_TEST else 5
EPOCHS = 2 if QUICK_TEST else 25
BATCH_SIZE = 32
LR = 3e-4
HORIZON_DAYS = 730           # 2-year survival endpoint (Hosny et al. 2018)
SLICES_PER_PATIENT = 5       # 2.5D crops sampled along the tumour (patient prediction = mean over crops)
CROP_PX = 128                # crops are cut from a 1 mm isotropic grid -> 128 mm field of view
CT_WIN = (-1000.0, 400.0)    # same lung window as Phase 1
BIN_WIDTH = 25               # HU bin width for texture features (pyradiomics default)
K_RAD = 10                   # radiomic features kept per fold (selected on training patients only)
MOD_DROP = 0.15              # modality dropout probability per branch (proposed model)
N_WORKERS = 2 * (os.cpu_count() or 1)   # DICOM reads are mostly I/O wait -> oversubscribe
INDEX_THREADS = 32                       # header-only reads when indexing the DICOM folders
SAVE_WEIGHTS = ["ct_cnn", "gated_md"]    # fold-0 weights kept for explainability


RES_DIR = f"{WORK_DIR}/phase3_results"
CROP_DIR = f"{RES_DIR}/crops_{FEATURE_SET}"   # separate cache per feature set, never mixed
WEIGHT_DIR = f"{RES_DIR}/weights"
FIG_DIR = f"{WORK_DIR}/figures/phase3"
CLINICAL_CSV_OUT = f"{RES_DIR}/clinical.csv"   # normalised clinical table (all patients)
INDEX_CSV = f"{RES_DIR}/dicom_index.csv"
COHORT_CSV = f"{RES_DIR}/cohort.csv"           # labelled patients with paired CT + RTSTRUCT
PREP_LOG_CSV = f"{RES_DIR}/preprocess_log.csv"


MODEL_NAMES = ["ct_cnn", "ct_rad_mid", "ct_rad_clin_mid", "gated", "gated_md"]
TAB_NAMES = ["tab_clinical_lr", "tab_volume_lr", "tab_radiomics_lr", "tab_clin_radiomics_lr", "tab_clin_radiomics_rf"]
LATE_NAMES = ["late_ct_rad", "late_ct_rad_clin"]
ORDER = TAB_NAMES + MODEL_NAMES + LATE_NAMES


for _d in (RES_DIR, CROP_DIR, WEIGHT_DIR, FIG_DIR):
   os.makedirs(_d, exist_ok=True)