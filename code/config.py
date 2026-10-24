"""Shared paths and settings for all Phase 1 scripts.


Defaults match Kaggle. To run elsewhere, set the environment variables, e.g.
   LUNA16_INPUT=/data/luna16 WORK_DIR=./work python 01_preprocess.py
"""
import os


INPUT_DIR = os.environ.get("LUNA16_INPUT", "/kaggle/input")      # searched for candidates*.csv and *.mhd
WORK_DIR = os.environ.get("WORK_DIR", "/kaggle/working")


PREP_DIR = f"{WORK_DIR}/preprocessed"          # .npy crops, labels.csv, labels_split.csv
OUT_DIR = f"{WORK_DIR}/model_results"          # checkpoints, histories, predictions, metrics
FIG_ROOT = f"{WORK_DIR}/figures"               # eda/, per_model/, comparison/, gradcam/


# preprocessing
CROP_SIZE = 64                 # pixels per side
CROP_MM = 50.0                 # physical side length of the patch (Setio et al. 2016)
HU_MIN, HU_MAX = -1000, 400    # lung window
NEGATIVE_TO_POSITIVE_RATIO = 3
RANDOM_SEED = 42


# split
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15


# training
MODELS = ["resnet50", "densenet121", "efficientnet_b0", "vgg16_bn"]
BATCH_SIZE = 32
NUM_EPOCHS = 25
LEARNING_RATE = 1e-4
INPUT_SIZE = 224               # ImageNet backbones are built for 224 px; 64 px leaves only a 2x2 final feature map
IMAGENET_MEAN, IMAGENET_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
SEED = 42
