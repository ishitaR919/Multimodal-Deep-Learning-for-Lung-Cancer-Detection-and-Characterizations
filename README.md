# Multimodal Deep Learning for Lung Cancer Detection and Characterization


Research Project-I · Department of Computer Science and Engineering · National Institute of Technology Rourkela ·
Autumn Semester 2026–27


**Student:** Ishita Rastogi (123CS0212) · **Supervisor:** Prof. Ratnakar Dash
**Area:** Medical Image Analysis, Multimodal Deep Learning, Computer Vision, Explainable AI


> Research prototype. Not intended for clinical use.


---


## 1. Problem statement


Lung cancer diagnosis draws on several kinds of information:


- **CT** shows the anatomy: the shape, size and texture of nodules and tumours.
- **PET** shows metabolic activity: how strongly tissue takes up the tracer.
- **Clinical data** adds age, sex, smoking history and TNM stage.
- **Radiomic features** are numbers computed from the tumour region: shape, intensity and texture.


Most AI models for lung cancer use only one of these, usually CT. This project asks:


1. Does combining modalities improve lung-cancer detection and characterization compared with a CT-only model?
2. Which fusion strategy works best: early, intermediate (feature-level), late, or attention-based / cross-modal?
3. How robust are multimodal models when one modality is missing at test time?
4. What do the models rely on? (explainability)


Models are judged on sensitivity, specificity, F1, ROC-AUC, PR-AUC, calibration (Brier score, ECE), bootstrap 95%
confidence intervals and, for survival, C-index and Kaplan–Meier analysis. Accuracy alone is not used.


## 2. Approach at a glance


No public dataset provides CT, PET, clinical, radiomic and outcome data for the same patients. Following the research
plan, the work is therefore organised in three phases. Each phase uses the dataset that contains the modalities its
question needs. **All unimodal-versus-multimodal comparisons are made within one dataset**, on the same patients and
the same folds. Results are never compared across datasets.


| Phase | Dataset | Modalities | Task | Unit |
|---|---|---|---|---|
| **1 · CT-only baseline** | LUNA16 (from LIDC-IDRI) | CT | Nodule vs non-nodule candidate | candidate |
| **2 · CT + PET fusion** | LUNG-PET-CT-Dx | CT, PET, clinical | Adenocarcinoma vs other subtypes | patient |
| **3 · CT + radiomics + clinical** | NSCLC-Radiomics (Lung1) | CT, tumour contour, radiomics, clinical, survival | 2-year overall survival | patient |


![Overall system architecture](figures/system_architecture.png)


Editable source: [`system_architecture.mmd`](system_architecture.mmd).


### Shared methodology (all phases)


- **Patient-level splitting.** No patient appears in both training and testing. Phase 1 uses one 70/15/15 split by
 scan; Phases 2–3 use stratified 5-fold cross-validation with a validation split inside each training fold.
- **Physically consistent preprocessing.** CT is lung-windowed (−1000 … 400 HU) and resampled to a fixed millimetre
 grid using the scan geometry. PET is converted to body-weight SUV and resampled onto the CT grid.
- **2.5D inputs.** Axial, coronal and sagittal planes through the lesion are stacked as 3 channels, so
 ImageNet-pretrained CNNs can be used.
- **Leakage-safe tabular pipelines.** Imputation, scaling and radiomic feature selection are fitted on training
 patients of each fold only.
- **Evaluation.** Sensitivity, specificity, precision, F1, ROC-AUC, PR-AUC, Brier score, ECE, reliability
 diagrams, patient-level bootstrap 95% CIs; for survival also C-index, Kaplan–Meier curves and log-rank tests.
- **Robustness.** Each fusion model is re-tested with one modality blanked out.
- **Explainability.** Grad-CAM heat maps, learned modality weights (gates), permutation importance and SHAP.


---


## 3. Phase 1 — CT-only baseline (LUNA16)


**Question.** How well can CT alone separate real nodules from false-positive candidates? Phase 1 builds and checks
the CT pipeline used later and gives a reference CT model.


**Data.** 888 thoracic CT scans; reference nodules are ≥ 3 mm and marked by at least 3 of 4 radiologists
(1,186 nodules). The candidate list (`candidates_V2.csv`) gives positions in millimetres and a 0/1 label.
The label means *nodule or not*, not benign vs malignant.


**Method.**
1. Keep all positive candidates plus 3 random negatives per positive (class balancing).
2. Lung window −1000 … 400 HU, scaled to 0 … 1 (Setio et al. 2016).
3. Resample a 50 mm cube around each candidate to 64³ voxels (0.78 mm isotropic).
4. Stack the central axial, coronal and sagittal planes as a 3 × 64 × 64 patch (2.5D; Roth et al. 2014).
5. Patient-level 70/15/15 split with a leakage check.
6. Fine-tune four ImageNet CNNs with identical settings: ResNet50, DenseNet121, EfficientNet-B0, VGG16-BN.
  Input upsampled to 224 px with ImageNet normalisation; class-weighted cross-entropy; Adam (lr 1e-4); 25 epochs;
  best checkpoint by validation ROC-AUC.
7. Hard and soft voting ensembles; Grad-CAM on the last convolutional block.


**Results (held-out test scans, threshold 0.5).**


| Model | Acc. | Sens. | Spec. | Prec. | F1 | ROC-AUC | PR-AUC | Brier ↓ | ECE ↓ |
|---|---|---|---|---|---|---|---|---|---|
| ResNet50 | 0.683 | 0.496 | **0.746** | 0.395 | 0.440 | 0.686 | 0.440 | 0.210 | 0.176 |
| DenseNet121 | 0.675 | 0.581 | 0.706 | 0.397 | 0.472 | 0.706 | 0.447 | 0.206 | **0.173** |
| EfficientNet-B0 | 0.550 | **0.800** | 0.466 | 0.333 | 0.471 | 0.699 | 0.441 | 0.223 | 0.222 |
| VGG16-BN | 0.666 | 0.608 | 0.686 | 0.392 | 0.477 | 0.709 | 0.460 | 0.215 | 0.197 |
| Hard voting | 0.649 | 0.642 | 0.651 | 0.380 | 0.478 | 0.695 | 0.391 | 0.258* | 0.227* |
| **Soft voting** | **0.697** | 0.608 | 0.727 | **0.426** | **0.501** | **0.720** | **0.484** | **0.202** | 0.179 |


\*Hard-vote scores are vote fractions (five values), so their Brier score and ECE are not directly comparable.
PR-AUC of a random classifier ≈ 0.25.


**Summary.** The four backbones lie within about 0.02 ROC-AUC of each other. Soft voting gives the best overall
result. Results are candidate-level on one split with sampled negatives, so they are not directly comparable with
the LUNA16 FROC/CPM leaderboard.


Details: [`phase1-ct-baseline/`](phase1-ct-baseline/).


---


## 4. Phase 2 — CT + PET fusion (LUNG-PET-CT-Dx)


**Question.** Does adding PET to CT improve histological subtype classification, and which fusion strategy works best?


**Data and cohort.** Patients with a CT and a PET from the same study, tumour bounding boxes and histology labels.
After checking that the annotated CT series belongs to the PET study (≥ 3 annotated slices located), the cohort is
**87 patients: 63 adenocarcinoma (label 0) and 24 other subtypes (label 1; 21 squamous, 3 small cell)**.


**Method.**
1. Pair CT and PET from the same study; use the CT series that carries the annotations.
2. Convert DICOM to CT (HU) and PET (body-weight SUV, decay-corrected; Kinahan & Fletcher 2010) volumes.
3. Resample CT and PET onto the same 160 mm / 128 px grid around the tumour box; up to 7 annotated slices per patient.
4. Simple tumour features (SUVmax/mean/std, HU mean/std, box size, z-extent) and clinical variables
  (sex, age, weight, smoking, T/N/M).
5. Models, all on the same stratified 5-fold patient-level CV:
  - tabular baselines: clinical, tumour features, both (logistic regression, random forest);
  - image baselines: CT only, PET only (ResNet-18);
  - early fusion (CT + PET as 2 channels), intermediate fusion (two encoders, concatenated features, optional
    clinical branch);
  - late fusion and soft/hard voting ensembles;
  - **proposed:** bidirectional cross-modal attention between CT and PET tokens with a learned modality gate, trained
    with **modality dropout** (Neverova et al. 2016), with and without clinical data.
6. Patient prediction = mean over slices. Missing-modality test, Grad-CAM, modality gates, SHAP.


**Results (pooled out-of-fold, patient level).**


| Model | ROC-AUC [95% CI] | PR-AUC | Bal. acc. | Sens. | Spec. | Brier |
|---|---|---|---|---|---|---|
| *to be filled from Phase 2 Cell 18* | | | | | | |


**Missing-modality robustness:** *to be filled from Phase 2 Cell 19.*
**Fusion vs CT only (paired bootstrap):** *to be filled from `compare_phases.py`.*


Details: [`phase2-ct-pet-fusion/`](phase2-ct-pet-fusion/).


---


## 5. Phase 3 — CT + radiomics + clinical survival model (NSCLC-Radiomics)


**Question.** Do radiomic and clinical data add value to a CT image model for predicting 2-year overall survival,
and is radiomics more informative than tumour volume alone?


**Data.** Planning CTs of NSCLC patients with the gross tumour volume (GTV-1) contour, clinical variables
(age, sex, T/N/M, overall stage, histology) and survival time with death status.
**Label:** 1 = died within 2 years, 0 = alive after 2 years. Patients censored before 2 years have no binary label
and are excluded from the classification task.


**Method.**
1. Pair each CT with its RTSTRUCT; rasterise the GTV-1 contour into a mask.
2. Resample a cube around the tumour to 1 mm isotropic voxels (IBSI-style; Zwanenburg et al. 2020).
3. Radiomic features with pyradiomics (shape, first-order, texture; van Griethuysen et al. 2017), or a built-in
  shape / first-order / GLCM set if pyradiomics is unavailable.
4. 2.5D CT crops (128 mm) through the tumour.
5. Models, all on the same stratified 5-fold patient-level CV:
  - tabular: clinical only, **tumour volume only** (checks whether radiomics is more than size; Welch et al. 2019),
    radiomics only, clinical + radiomics (early fusion), random forest;
  - CT CNN (ResNet-18);
  - intermediate fusion: CT + radiomics, CT + radiomics + clinical;
  - **proposed:** gated attention fusion of CT, radiomic and clinical embeddings, trained with modality dropout;
  - late fusion.
6. Radiomic feature selection (variance → correlation filter → top-k) fitted inside each training fold.
7. Evaluation adds C-index and Kaplan–Meier curves of predicted high- vs low-risk groups (log-rank test).


**Results (pooled out-of-fold, patient level).**


| Model | ROC-AUC [95% CI] | PR-AUC | C-index | Sens. | Spec. | Brier |
|---|---|---|---|---|---|---|
| *to be filled from Phase 3 Cell 13* | | | | | | |


**Missing-modality robustness:** *to be filled from Phase 3 Cell 14.*


Details: [`phase3-radiomics-survival/`](phase3-radiomics-survival/).


---


## 6. What the project delivers


1. A **CT-only baseline** with a documented, leakage-safe pipeline (Phase 1).
2. A **controlled comparison of fusion strategies** (early, intermediate, late, attention-based) for CT + PET
  (Phase 2) and CT + radiomics + clinical (Phase 3), each against its own unimodal baselines.
3. A **proposed fusion model** (attention / gated fusion with modality dropout) with ablations showing the
  contribution of each modality and component.
4. **Missing-modality robustness** results.
5. **Explainability** outputs: Grad-CAM, learned modality weights, permutation importance, SHAP.
6. **Reproducible code** (Kaggle notebooks and step-by-step scripts) and a **Streamlit demo** to upload a CT,
  view slices, run the Phase 1 models with Grad-CAM, and browse all results.


## 7. Repository structure


```
.
├── README.md                         # this file
├── system_architecture.mmd           # overall architecture (Mermaid)
├── figures/system_architecture.png   # rendered diagram
├── phase1-ct-baseline/               # Phase 1: notebook, scripts, README, figures, results
├── phase2-ct-pet-fusion/             # Phase 2: notebook, scripts, README
├── phase3-radiomics-survival/        # Phase 3: notebook, scripts, README
├── compare_phases.py                 # Phase 2 fusion-vs-CT paired bootstrap + per-phase summary table
├── streamlit_ui/                     # research demo app
└── report/                           # LaTeX reports
```


No patient data, model weights or intermediate volumes are stored in this repository.


## 8. How to run


Each phase runs as one Kaggle notebook (GPU and Internet on). Use **Save Version → Save & Run All** so long runs
continue after the browser is closed, then download the results bundle from the notebook output.


| Phase | Attach on Kaggle | Notebook | Output |
|---|---|---|---|
| 1 | LUNA16 (with `subset*` folders, `candidates_V2.csv`, `annotations.csv`) | `phase1-ct-baseline/kaggle_notebook/phase1_luna16.py` | `phase1_results_bundle.zip` |
| 2 | [`rangan2510/lung-pet-ct-dx`](https://www.kaggle.com/datasets/rangan2510/lung-pet-ct-dx) | `phase2-ct-pet-fusion/kaggle_notebook/phase2.py` | `phase2_results_bundle.zip` |
| 3 | [`umutkrdrms/nsclc-radiomics`](https://www.kaggle.com/datasets/umutkrdrms/nsclc-radiomics) + Lung1 clinical CSV | `phase3-radiomics-survival/kaggle_notebook/phase3.py` | `phase3_results_bundle.zip` |


Each phase folder also has a `code/` directory with the same pipeline split into numbered scripts, and its own
README with details.


Local comparison and demo:


```bash
python compare_phases.py --p1 phase1_results_bundle --p2 phase2_results_bundle --out comparison
cd streamlit_ui && pip install -r requirements.txt && streamlit run app.py
```


## 9. Datasets


| Dataset | Source | Citation |
|---|---|---|
| LUNA16 | <https://luna16.grand-challenge.org/> | Setio et al. 2017; Armato et al. 2011 (LIDC-IDRI) |
| LUNG-PET-CT-Dx | <https://www.cancerimagingarchive.net/collection/lung-pet-ct-dx/> | Li et al. 2020 |
| NSCLC-Radiomics (Lung1) | <https://www.cancerimagingarchive.net/collection/nsclc-radiomics/> | Aerts et al. 2014 |


Please follow each dataset's terms of use.


## 10. Limitations


- Each phase uses a different public dataset, so results are compared within a dataset, not across phases.
- Phase 1 reports candidate-level metrics on one split, not the LUNA16 FROC/CPM protocol.
- Phase 2 has a small cohort (87 patients); confidence intervals are wide.
- Phase 3 excludes patients censored before 2 years from the binary task.
- No external validation set; all results are internal.


## 11. References


- S. G. Armato III *et al.*, "The Lung Image Database Consortium (LIDC) and Image Database Resource Initiative (IDRI)," *Medical Physics*, 38(2), 2011.
- A. A. A. Setio *et al.*, "Validation, comparison, and combination of algorithms for automatic detection of pulmonary nodules in CT images: the LUNA16 challenge," *Medical Image Analysis*, 42, 2017.
- A. A. A. Setio *et al.*, "Pulmonary nodule detection in CT images: false positive reduction using multi-view convolutional networks," *IEEE TMI*, 35(5), 2016.
- H. R. Roth *et al.*, "A new 2.5D representation for lymph node detection using random sets of deep convolutional neural network observations," *MICCAI*, 2014.
- P. Li *et al.*, "A large-scale CT and PET/CT dataset for lung cancer diagnosis (Lung-PET-CT-Dx)," The Cancer Imaging Archive, 2020.
- H. J. W. L. Aerts *et al.*, "Decoding tumour phenotype by noninvasive imaging using a quantitative radiomics approach," *Nature Communications*, 5, 2014.
- A. Hosny *et al.*, "Deep learning for lung cancer prognostication: a retrospective multi-cohort radiomics study," *PLoS Medicine*, 15(11), 2018.
- M. L. Welch *et al.*, "Vulnerabilities of radiomic signature development: the need for safeguards," *Radiotherapy and Oncology*, 130, 2019.
- J. J. M. van Griethuysen *et al.*, "Computational radiomics system to decode the radiographic phenotype," *Cancer Research*, 77(21), 2017.
- A. Zwanenburg *et al.*, "The Image Biomarker Standardization Initiative," *Radiology*, 295(2), 2020.
- P. E. Kinahan and J. W. Fletcher, "PET/CT standardized uptake values in clinical practice and assessing response to therapy," *Seminars in Ultrasound, CT and MRI*, 31(6), 2010.
- S.-C. Huang *et al.*, "Fusion of medical imaging and electronic health records using deep learning: a systematic review," *npj Digital Medicine*, 3, 2020.
- N. Neverova *et al.*, "ModDrop: adaptive multi-modal gesture recognition," *IEEE TPAMI*, 38(8), 2016.
- A. Vaswani *et al.*, "Attention is all you need," *NeurIPS*, 2017.
- H.-C. Shin *et al.*, "Deep convolutional neural networks for computer-aided detection," *IEEE TMI*, 35(5), 2016.
- N. Tajbakhsh *et al.*, "Convolutional neural networks for medical image analysis: full training or fine tuning?" *IEEE TMI*, 35(5), 2016.
- K. He *et al.*, "Deep residual learning for image recognition," *CVPR*, 2016.
- G. Huang *et al.*, "Densely connected convolutional networks," *CVPR*, 2017.
- M. Tan and Q. V. Le, "EfficientNet," *ICML*, 2019.
- K. Simonyan and A. Zisserman, "Very deep convolutional networks for large-scale image recognition," *ICLR*, 2015.
- E. Yagis *et al.*, "Effect of data leakage in brain MRI classification using 2D convolutional neural networks," *Scientific Reports*, 11, 2021.
- T. Saito and M. Rehmsmeier, "The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets," *PLoS ONE*, 10(3), 2015.
- C. Guo *et al.*, "On calibration of modern neural networks," *ICML*, 2017.
- F. E. Harrell *et al.*, "Evaluating the yield of medical tests," *JAMA*, 247(18), 1982.
- E. L. Kaplan and P. Meier, "Nonparametric estimation from incomplete observations," *JASA*, 53(282), 1958.
- R. R. Selvaraju *et al.*, "Grad-CAM: visual explanations from deep networks via gradient-based localization," *ICCV*, 2017.
- S. M. Lundberg and S.-I. Lee, "A unified approach to interpreting model predictions," *NeurIPS*, 2017.