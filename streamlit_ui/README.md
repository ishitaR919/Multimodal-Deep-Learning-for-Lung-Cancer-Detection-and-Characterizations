# Streamlit research demo


A local web UI for the project:


| Page | What it does |
|---|---|
| **Overview** | Project summary, Phase 1 architecture diagram, which result folders were found |
| **Phase 1 - Predict** | Upload a CT scan (`.mhd` + `.raw`) and a candidate position, or a ready `.npy` patch. The four trained CNNs and their soft vote give the nodule probability; Grad-CAM shows where each model looked. |
| **Phase 1 - Results** | Test-set metrics table and all Phase 1 figures |
| **Phase 2 / Phase 3 - Results** | Results tables, missing-modality table and figures from the Kaggle bundles |


The app is self-contained: it repeats the Phase 1 preprocessing (lung window, 50 mm three-view patch) and model
input (224 px, ImageNet normalisation) exactly, and only reads your trained `*_best.pt` checkpoints. Nothing in the
Phase 1 code or results is changed. Research prototype only — not for clinical use.


## Setup


```bash
cd streamlit_ui
pip install -r requirements.txt
streamlit run app.py
```


Expected folders (all paths can be changed in the sidebar under **Artefact folders**):


```
RP/
├── phase1-ct-baseline/
│   ├── model_results/
│   │   ├── resnet50_best.pt          # <- your downloaded checkpoints (or any folder set in the sidebar)
│   │   ├── densenet121_best.pt
│   │   ├── efficientnet_b0_best.pt
│   │   ├── vgg16_bn_best.pt
│   │   └── *.csv
│   └── figures/                      # incl. flow_diagram.png
├── phase2_results_bundle/            # unzipped phase2_results_bundle.zip
│   ├── phase2_results/
│   └── figures_phase2/
├── phase3_results_bundle/            # unzipped phase3_results_bundle.zip
│   ├── phase3_results/
│   └── figures_phase3/
└── streamlit_ui/app.py
```


### Checkpoints


Point **Artefact folders → Folder with the downloaded \*_best.pt checkpoints** at wherever you saved
`resnet50_best.pt`, `densenet121_best.pt`, `efficientnet_b0_best.pt` and `vgg16_bn_best.pt`. Any subset works;
missing models are skipped.


## Using the predict page


- **CT scan:** upload a LUNA16 `.mhd` and its `.raw` together (keep the original file names). Optionally upload
 `candidates_V2.csv` or `annotations.csv` to choose a candidate of that scan from a list; otherwise type the
 position in millimetres (world coordinates, as in the LUNA16 CSVs).
- **Patch:** upload any `.npy` file produced by the Phase 1 preprocessing step (`3 × 64 × 64`).


The result is the probability that the candidate is a **nodule**, not whether it is malignant.



