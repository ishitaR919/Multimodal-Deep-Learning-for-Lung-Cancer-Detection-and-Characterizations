"""Step 10 - explainability: Grad-CAM on the CT / PET branches of the saved fold-0 models, the learned modality
gate of the proposed model, and permutation / SHAP importance of the tabular features (descriptive only).


Usage:  python 10_explain.py


Input:  crops (step 3), WEIGHT_DIR/<model>_fold0.pt (step 6)
Output (FIG_DIR): explain_gradcam_<model>.png, explain_modality_gate.png, explain_tabular_importance.png,
                 explain_tabular_shap.png
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance


from config import SEED, PET_SUV_MAX, FIG_DIR, CLIN_COLS, RAD_COLS
from common import load_state
from models import DEVICE, grad_cam, load_fold0


CAM_LAYERS = {"early": lambda m: (m.enc.layer4, m.enc.layer4),
             "intermediate": lambda m: (m.enc_ct.layer4, m.enc_pet.layer4),
             "cross_attn_md": lambda m: (m.enc_ct.layer4, m.enc_pet.layer4)}




def main():
   st = load_state()
   fold0_test = [p for p in st.pids if st.fold_of[p] == 0]
   chosen = [p for lab in (0, 1) for p in [q for q in fold0_test if st.labels[q] == lab][:3]]
   for name, get_layers in CAM_LAYERS.items():
       model = load_fold0(name)
       if model is None or not chosen:
           print(f"Grad-CAM skipped for {name} (no fold-0 weights; train it first)"); continue
       fig, axes = plt.subplots(len(chosen), 4, figsize=(11, 2.8 * len(chosen))); axes = np.atleast_2d(axes)
       for i, pid in enumerate(chosen):
           ct, pet = st.crops[pid]; j = ct.shape[0] // 2
           x = torch.from_numpy(np.stack([ct[j], pet[j]])[None]).float().to(DEVICE)
           with torch.no_grad():
               prob = torch.softmax(model(x, None), 1)[0, 1].item()
           cls = int(prob >= 0.5)
           l_ct, l_pet = get_layers(model)
           cam_ct, cam_pet = grad_cam(model, x, None, l_ct, cls), grad_cam(model, x, None, l_pet, cls)
           axes[i, 0].imshow(ct[j], cmap="gray", vmin=0, vmax=1); axes[i, 1].imshow(pet[j] * PET_SUV_MAX, cmap="hot", vmin=0, vmax=PET_SUV_MAX)
           axes[i, 2].imshow(ct[j], cmap="gray", vmin=0, vmax=1); axes[i, 2].imshow(cam_ct, cmap="jet", alpha=0.4)
           axes[i, 3].imshow(pet[j] * PET_SUV_MAX, cmap="gray"); axes[i, 3].imshow(cam_pet, cmap="jet", alpha=0.4)
           axes[i, 0].set_ylabel(f"{pid}\ntrue={st.labels[pid]} p={prob:.2f}", fontsize=8)
           for a in axes[i]:
               a.set_xticks([]); a.set_yticks([])
       shared = l_ct is l_pet                                  # early fusion: one encoder sees both channels
       for a, t in zip(axes[0], ["CT", "PET (SUV)", "Grad-CAM on CT" if shared else "Grad-CAM on CT branch",
                                 "Grad-CAM on PET (same map)" if shared else "Grad-CAM on PET branch"]):
           a.set_title(t, fontsize=9)
       plt.suptitle(f"{name}: Grad-CAM for the predicted class (fold-0 held-out patients, middle slice)"); plt.tight_layout()
       plt.savefig(f"{FIG_DIR}/explain_gradcam_{name}.png", dpi=140); plt.close()


   gate_model = load_fold0("cross_attn_md")
   if gate_model is not None:
       rows = []
       with torch.no_grad():
           for pid in fold0_test:
               ct, pet = st.crops[pid]
               gate_model(torch.from_numpy(np.stack([ct, pet], 1)).float().to(DEVICE), None)
               g = gate_model.last_gate.mean(0).cpu().numpy()
               rows.append({"patient_id": pid, "label": st.labels[pid], "w_CT": g[0], "w_PET": g[1]})
       G = pd.DataFrame(rows).groupby("label")[["w_CT", "w_PET"]].mean()
       print("Mean learned modality weights by class (fold-0 held-out patients):"); print(G.round(3).to_string())
       G.plot(kind="bar", figsize=(5, 3.5), title="Cross-attention model: learned modality gate"); plt.ylabel("weight"); plt.tight_layout()
       plt.savefig(f"{FIG_DIR}/explain_modality_gate.png", dpi=140); plt.close()


   # Tabular importance (descriptive: model fit on all patients, NOT an evaluation)
   cols = CLIN_COLS + RAD_COLS
   Xt = pd.DataFrame(SimpleImputer(strategy="median", keep_empty_features=True).fit_transform(st.master.set_index("patient_id").loc[st.pids, cols]), columns=cols)
   yt = np.array([st.labels[p] for p in st.pids])
   rf = RandomForestClassifier(n_estimators=500, min_samples_leaf=2, class_weight="balanced_subsample", random_state=SEED).fit(Xt, yt)
   pi_ = permutation_importance(rf, Xt, yt, n_repeats=20, random_state=SEED, scoring="roc_auc")
   imp = pd.Series(pi_.importances_mean, index=cols).sort_values()
   plt.figure(figsize=(6, 5)); imp.plot(kind="barh", color="#4C72B0"); plt.title("Permutation importance (RF, all patients; descriptive)")
   plt.tight_layout(); plt.savefig(f"{FIG_DIR}/explain_tabular_importance.png", dpi=140); plt.close()
   try:
       import shap
       sv = shap.TreeExplainer(rf).shap_values(Xt)
       sv = sv[1] if isinstance(sv, list) else (sv[:, :, 1] if np.ndim(sv) == 3 else sv)
       shap.summary_plot(sv, Xt, show=False); plt.tight_layout()
       plt.savefig(f"{FIG_DIR}/explain_tabular_shap.png", dpi=140, bbox_inches="tight"); plt.close()
   except Exception as e:
       print("SHAP summary skipped:", type(e).__name__)




if __name__ == "__main__":
   main()



