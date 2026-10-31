"""Research demo UI for the multimodal lung-cancer project.


Pages
- Overview            project summary, architecture diagram, which artefacts were found
- Phase 1 - Predict   upload a CT (.mhd + .raw + candidate position) or a ready 3x64x64 .npy patch,
                     run the four trained CNNs + soft voting, show the three input views and Grad-CAM
- Phase 1 - Results   test-set metrics table and all saved figures
- Phase 2 / 3 Results results tables and figures from the downloaded Kaggle bundles (shown if present)


Run:  streamlit run app.py
Research prototype only - not for clinical use.
"""
import os
import tempfile
from pathlib import Path


import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models as tvm, transforms


APP_DIR = Path(__file__).resolve().parent
RP_DIR = APP_DIR.parent


st.set_page_config(page_title="Multimodal Lung Cancer AI - Research Demo", page_icon="🫁", layout="wide")


# --------------------------------------------------------------------------- sidebar: where the artefacts are
st.sidebar.title("🫁 Lung Cancer AI")
page = st.sidebar.radio("Page", ["Overview", "Phase 1 - Predict", "Phase 1 - Results", "Phase 2 - Results",
                                "Phase 3 - Results"])
with st.sidebar.expander("Artefact folders", expanded=False):
   P1_DIR = Path(st.text_input("Phase 1 folder (model_results/, figures/)",
                               os.environ.get("PHASE1_DIR", str(RP_DIR / "phase1-ct-baseline"))))
   CKPT_DIR = Path(st.text_input("Folder with the downloaded *_best.pt checkpoints",
                                 os.environ.get("CKPT_DIR", str(P1_DIR / "model_results"))))
   P2_DIR = Path(st.text_input("Phase 2 bundle (phase2_results/, figures_phase2/)",
                               os.environ.get("PHASE2_DIR", str(RP_DIR / "phase2_results_bundle"))))
   P3_DIR = Path(st.text_input("Phase 3 bundle (phase3_results/, figures_phase3/)",
                               os.environ.get("PHASE3_DIR", str(RP_DIR / "phase3_results_bundle"))))
st.sidebar.caption("Research prototype - not a medical device. Do not use for diagnosis.")


MODEL_LABELS = {"resnet50": "ResNet50", "densenet121": "DenseNet121", "efficientnet_b0": "EfficientNet-B0",
               "vgg16_bn": "VGG16-BN"}
# last convolutional block of each backbone (same choice as the Phase 1 Grad-CAM script)
TARGET_LAYERS = {"resnet50": lambda m: [m.layer4[-1]], "densenet121": lambda m: [m.features[-1]],
                "efficientnet_b0": lambda m: [m.features[-1]], "vgg16_bn": lambda m: [m.features[-1]]}




# --------------------------------------------------------------------------- Phase 1 pipeline (same as training)
CROP_SIZE, CROP_MM = 64, 50.0              # 50 mm cube -> 64 px (Setio et al. 2016)
HU_MIN, HU_MAX = -1000, 400                # lung window
INPUT_SIZE = 224
IMAGENET_MEAN, IMAGENET_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
to_model_input = transforms.Compose([transforms.Resize((INPUT_SIZE, INPUT_SIZE), antialias=True),
                                    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])




def load_ct_volume(mhd_path):
   """Load a .mhd/.raw CT scan, clip to the lung window and scale to [0, 1] (geometry kept)."""
   import SimpleITK as sitk
   img = sitk.ReadImage(mhd_path, sitk.sitkFloat32)
   img = sitk.Clamp(img, sitk.sitkFloat32, HU_MIN, HU_MAX)
   return (img - HU_MIN) / (HU_MAX - HU_MIN)




def extract_25d_crop(img, world_xyz, crop_size=CROP_SIZE, crop_mm=CROP_MM):
   """Isotropic crop_mm cube around the candidate -> central axial / coronal / sagittal planes, (3, 64, 64)."""
   import SimpleITK as sitk
   sp = crop_mm / crop_size
   D = np.array(img.GetDirection()).reshape(3, 3)
   ref = sitk.Image(crop_size, crop_size, crop_size, sitk.sitkFloat32)
   ref.SetSpacing((sp, sp, sp))
   ref.SetDirection(img.GetDirection())
   ref.SetOrigin(tuple(float(v) for v in np.asarray(world_xyz) - D @ np.full(3, sp * (crop_size - 1) / 2.0)))
   cube = sitk.GetArrayFromImage(sitk.Resample(img, ref, sitk.Transform(), sitk.sitkLinear, 0.0))  # z, y, x
   c = crop_size // 2
   return np.stack([cube[c], cube[:, c, :], cube[:, :, c]]).astype(np.float32)




def build_model(name):
   """Same architectures and 2-class heads as training; weights come from the checkpoint (no download)."""
   if name == "resnet50":
       m = tvm.resnet50(weights=None); m.fc = nn.Linear(m.fc.in_features, 2)
   elif name == "densenet121":
       m = tvm.densenet121(weights=None); m.classifier = nn.Linear(m.classifier.in_features, 2)
   elif name == "efficientnet_b0":
       m = tvm.efficientnet_b0(weights=None); m.classifier[1] = nn.Linear(m.classifier[1].in_features, 2)
   elif name == "vgg16_bn":
       m = tvm.vgg16_bn(weights=None); m.classifier[6] = nn.Linear(m.classifier[6].in_features, 2)
   else:
       raise ValueError(name)
   return m.to(DEVICE)




@st.cache_resource(show_spinner="Loading trained Phase 1 models...")
def load_models(ckpt_dir):
   loaded = {}
   for name in MODEL_LABELS:
       ckpt = Path(ckpt_dir) / f"{name}_best.pt"
       if ckpt.exists():
           m = build_model(name)
           m.load_state_dict(torch.load(ckpt, map_location=DEVICE))
           loaded[name] = m.eval()
   return loaded




def predict_all(models, crop):
   x = to_model_input(torch.from_numpy(crop).float()).unsqueeze(0).to(DEVICE)
   with torch.no_grad():
       probs = {n: float(torch.softmax(m(x), 1)[0, 1]) for n, m in models.items()}
   return x, probs




def gradcam_overlay(model, name, x, crop, target_class):
   from pytorch_grad_cam import GradCAM
   from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
   from pytorch_grad_cam.utils.image import show_cam_on_image
   with GradCAM(model=model, target_layers=TARGET_LAYERS[name](model)) as cam:
       g = cam(input_tensor=x, targets=[ClassifierOutputTarget(target_class)])[0]
   axial = F.interpolate(torch.from_numpy(crop[0:1][None]).float(), size=g.shape, mode="bilinear",
                         align_corners=False)[0, 0].clamp(0, 1).numpy()
   return show_cam_on_image(np.stack([axial] * 3, -1).astype(np.float32), g, use_rgb=True)




def show_views(crop):
   """Axial / coronal / sagittal channels. Coronal and sagittal are flipped so the patient's head is up."""
   cols = st.columns(3)
   for c, img, title in zip(cols, [crop[0], np.flipud(crop[1]), np.flipud(crop[2])], ["Axial", "Coronal", "Sagittal"]):
       c.image(np.clip(img, 0, 1), caption=title, use_container_width=True, clamp=True)




# --------------------------------------------------------------------------- shared helpers
def show_figures(fig_dir, title=None):
   files = sorted(Path(fig_dir).glob("*.png")) if Path(fig_dir).exists() else []
   if not files:
       return False
   if title:
       st.subheader(title)
   for i in range(0, len(files), 2):
       cols = st.columns(2)
       for c, f in zip(cols, files[i:i + 2]):
           c.image(str(f), caption=f.stem.replace("_", " "), use_container_width=True)
   return True




def show_phase_results(bundle, res_sub, fig_sub, title, baseline, note):
   st.title(title)
   st.caption(note)
   res_csv = bundle / res_sub / "results_all_models.csv"
   if not res_csv.exists():
       st.info(f"No results found at `{res_csv}`. Unzip the Kaggle bundle there or change the folder in the sidebar.")
       return
   res = pd.read_csv(res_csv, index_col=0)
   cols = [c for c in ["roc_auc", "roc_auc_lo", "roc_auc_hi", "pr_auc", "c_index", "balanced_acc", "sensitivity",
                       "specificity", "f1", "brier", "n_patients"] if c in res.columns]
   best = res["roc_auc"].astype(float).idxmax()
   m1, m2, m3 = st.columns(3)
   m1.metric("Best model (ROC-AUC)", best, f"{float(res.loc[best, 'roc_auc']):.3f}")
   if baseline in res.index:
       m2.metric(f"Baseline: {baseline}", f"{float(res.loc[baseline, 'roc_auc']):.3f}",
                 f"{float(res.loc[best, 'roc_auc']) - float(res.loc[baseline, 'roc_auc']):+.3f} vs best")
   m3.metric("Patients", int(res["n_patients"].max()) if "n_patients" in res else "-")
   st.dataframe(res[cols].astype(float).round(3).style.highlight_max(subset=["roc_auc", "pr_auc"], color="#d4edda"),
                use_container_width=True)
   miss = bundle / res_sub / "results_missing_modality.csv"
   if miss.exists():
       st.subheader("Missing-modality robustness (ROC-AUC)")
       st.dataframe(pd.read_csv(miss).pivot(index="model", columns="available", values="roc_auc").round(3),
                    use_container_width=True)
   show_figures(bundle / fig_sub, "Figures")




# =========================================================================== pages
if page == "Overview":
   st.title("Multimodal Deep Learning for Lung Cancer Detection and Characterization")
   st.markdown("Research Project-I, Department of Computer Science and Engineering, NIT Rourkela - Autumn 2026-27")
   c1, c2, c3 = st.columns(3)
   c1.markdown("#### Phase 1 - CT baseline\nLUNA16 · nodule vs non-nodule candidate · 2.5D three-view patches · "
               "ResNet50 / DenseNet121 / EfficientNet-B0 / VGG16-BN + voting")
   c2.markdown("#### Phase 2 - CT + PET fusion\nLUNG-PET-CT-Dx · adenocarcinoma vs other subtypes · early / "
               "intermediate / late / cross-attention fusion · modality dropout")
   c3.markdown("#### Phase 3 - CT + radiomics + clinical\nNSCLC-Radiomics (Lung1) · 2-year survival · radiomic "
               "signature · gated fusion · Kaplan-Meier risk groups")
   flow = P1_DIR / "figures" / "flow_diagram.png"
   if flow.exists():
       st.image(str(flow), caption="Phase 1 architecture", use_container_width=True)
   st.subheader("Artefacts found")
   status = {
       "Phase 1 trained models (*.pt)": any(CKPT_DIR.glob("*_best.pt")),
       "Phase 1 results": (P1_DIR / "model_results" / "final_metrics_all_models_and_ensembles.csv").exists(),
       "Phase 2 results": (P2_DIR / "phase2_results" / "results_all_models.csv").exists(),
       "Phase 3 results": (P3_DIR / "phase3_results" / "results_all_models.csv").exists(),
   }
   st.table(pd.DataFrame({"available": ["✅" if v else "—" for v in status.values()]}, index=list(status)))


elif page == "Phase 1 - Predict":
   st.title("Phase 1 - Is this candidate a nodule?")
   st.caption("Output = probability that the candidate is a real nodule (LUNA16 false-positive reduction). "
              "It is not a malignancy prediction.")
   try:
       models = load_models(str(CKPT_DIR))
   except Exception as e:
       st.error(f"Could not load the checkpoints: {e}")
       st.stop()
   if not models:
       st.warning(f"No `<model>_best.pt` files in `{CKPT_DIR}`. Set the checkpoint folder in the sidebar "
                  "(Artefact folders).")
       st.stop()
   st.success(f"Loaded models: {', '.join(MODEL_LABELS[m] for m in models)}  ·  device: {DEVICE}")


   mode = st.radio("Input", ["CT scan (.mhd + .raw) + candidate position", "Preprocessed patch (.npy, 3×64×64)"],
                   horizontal=True)
   crop = None
   if mode.startswith("CT scan"):
       files = st.file_uploader("Upload the .mhd header and its .raw file together", type=["mhd", "raw", "zraw"],
                                accept_multiple_files=True)
       cand_csv = st.file_uploader("Optional: candidates_V2.csv / annotations.csv to pick a position", type=["csv"])
       mhd = next((f for f in files or [] if f.name.endswith(".mhd")), None)
       if mhd and len(files) >= 2:
           tmp = Path(tempfile.mkdtemp())
           for f in files:                                   # keep original names: the .mhd points to the .raw by name
               (tmp / f.name).write_bytes(f.getbuffer())
           series_uid = Path(mhd.name).stem
           x0 = y0 = z0 = 0.0
           if cand_csv is not None:
               cdf = pd.read_csv(cand_csv)
               rows = cdf[cdf["seriesuid"] == series_uid] if "seriesuid" in cdf else cdf.iloc[0:0]
               if len(rows):
                   label_col = "class" if "class" in rows else None
                   opts = [f"#{i}: ({r.coordX:.1f}, {r.coordY:.1f}, {r.coordZ:.1f})"
                           + (f"  label={int(r[label_col])}" if label_col else "") for i, r in rows.iterrows()]
                   pick = st.selectbox(f"{len(rows)} candidates for this scan", range(len(rows)), format_func=lambda k: opts[k])
                   x0, y0, z0 = rows.iloc[pick][["coordX", "coordY", "coordZ"]].astype(float)
               else:
                   st.info("No rows in the CSV for this scan - enter the position manually.")
           c1, c2, c3 = st.columns(3)
           wx = c1.number_input("coordX (mm)", value=float(x0), format="%.2f")
           wy = c2.number_input("coordY (mm)", value=float(y0), format="%.2f")
           wz = c3.number_input("coordZ (mm)", value=float(z0), format="%.2f")
           if st.button("Extract patch and predict", type="primary"):
               with st.spinner("Reading scan and resampling a 50 mm cube..."):
                   img = load_ct_volume(str(tmp / mhd.name))
                   crop = extract_25d_crop(img, (wx, wy, wz))
       elif files:
           st.info("Upload both the .mhd and the matching .raw file.")
   else:
       up = st.file_uploader("Upload a patch saved by the preprocessing step", type=["npy"])
       if up is not None:
           arr = np.load(up)
           if arr.shape != (3, CROP_SIZE, CROP_SIZE):
               st.error(f"Expected shape (3, {CROP_SIZE}, {CROP_SIZE}), got {arr.shape}")
           else:
               crop = arr.astype(np.float32)


   if crop is not None:
       st.subheader("Input: three orthogonal 50 mm views")
       show_views(crop)
       x, probs = predict_all(models, crop)
       soft = float(np.mean(list(probs.values())))
       votes = sum(p >= 0.5 for p in probs.values())


       st.subheader("Prediction")
       k1, k2, k3 = st.columns(3)
       k1.metric("Soft-vote nodule probability", f"{soft:.1%}")
       k2.metric("Soft-vote decision (≥ 50 %)", "Nodule" if soft >= 0.5 else "Not a nodule")
       k3.metric("Models voting 'nodule'", f"{votes} / {len(probs)}")
       st.bar_chart(pd.DataFrame({"nodule probability": {MODEL_LABELS[n]: p for n, p in probs.items()}}), height=260)


       st.subheader("Grad-CAM - where the model looked")
       name = st.selectbox("Model", list(models), format_func=MODEL_LABELS.get)
       cls = int(probs[name] >= 0.5)
       try:
           overlay = gradcam_overlay(models[name], name, x, crop, cls)
           c1, c2 = st.columns(2)
           c1.image(np.clip(crop[0], 0, 1), caption="Axial view", use_container_width=True, clamp=True)
           c2.image(overlay, caption=f"Grad-CAM for '{'nodule' if cls else 'not a nodule'}' ({MODEL_LABELS[name]})",
                    use_container_width=True)
       except ImportError:
           st.info("Install `grad-cam` to see heat maps: pip install grad-cam")


elif page == "Phase 1 - Results":
   st.title("Phase 1 - CT-only baseline results (LUNA16 test scans)")
   csv = P1_DIR / "model_results" / "final_metrics_all_models_and_ensembles.csv"
   if not csv.exists():
       st.info(f"No results at `{csv}`.")
   else:
       res = pd.read_csv(csv, index_col=0).astype(float)
       best = res["roc_auc"].idxmax()
       m1, m2, m3 = st.columns(3)
       m1.metric("Best ROC-AUC", f"{res.loc[best, 'roc_auc']:.3f}", best)
       m2.metric("Best PR-AUC", f"{res['pr_auc'].max():.3f}", res["pr_auc"].idxmax())
       m3.metric("Highest sensitivity", f"{res['recall_sensitivity'].max():.3f}", res["recall_sensitivity"].idxmax())
       st.dataframe(res.round(3).style.highlight_max(subset=["roc_auc", "pr_auc", "f1"], color="#d4edda"),
                    use_container_width=True)
   tabs = st.tabs(["Comparison", "Per model", "Grad-CAM", "EDA"])
   for tab, sub in zip(tabs, ["comparison", "per_model", "gradcam", "eda"]):
       with tab:
           if not show_figures(P1_DIR / "figures" / sub):
               st.info(f"No figures in `{P1_DIR / 'figures' / sub}`.")


elif page == "Phase 2 - Results":
   show_phase_results(P2_DIR, "phase2_results", "figures_phase2", "Phase 2 - CT + PET fusion (LUNG-PET-CT-Dx)",
                      "ct_only", "Adenocarcinoma vs other subtypes; pooled out-of-fold patient-level results from "
                                 "5-fold cross-validation with 95% bootstrap CIs.")


else:
   show_phase_results(P3_DIR, "phase3_results", "figures_phase3",
                      "Phase 3 - CT + radiomics + clinical (NSCLC-Radiomics)", "ct_cnn",
                      "2-year overall survival; pooled out-of-fold patient-level results from 5-fold cross-validation "
                      "with 95% bootstrap CIs and C-index.")