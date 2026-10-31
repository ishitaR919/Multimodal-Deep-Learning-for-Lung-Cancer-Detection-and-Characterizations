"""Step 9 - explainability: Grad-CAM (fold-0 models), learned modality gate, radiomic signature,
permutation importance and SHAP for clinical + signature features.


The signature / importance / SHAP models are fitted on all patients and are descriptive, not an evaluation.


Usage:   python 09_explain.py
Inputs:  RES_DIR/cohort.csv, CROP_DIR/*.npz, WEIGHT_DIR/<model>_fold0.pt (step 5)
Outputs: FIG_DIR/explain_gradcam_<model>.png, explain_modality_gate.png, explain_radiomic_signature.png,
        explain_tabular_importance.png, explain_tabular_shap.png
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402


from config import FIG_DIR, SAVE_WEIGHTS, SEED  # noqa: E402
from common import load_state, torch_parts, rad_steps  # noqa: E402




def grad_cam(T, model, x, r, c, layer, class_idx):
   F, store = T["F"], {}


   def fwd_hook(module, inp, out):
       store["a"] = out
       out.register_hook(lambda g: store.__setitem__("g", g))


   handle = layer.register_forward_hook(fwd_hook)
   model.zero_grad()
   with T["torch"].enable_grad():
       model(x, r, c)[0, class_idx].backward()
   handle.remove()
   cam = F.relu((store["g"].mean(dim=(2, 3), keepdim=True) * store["a"]).sum(1, keepdim=True))
   cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
   cam = cam - cam.min()
   return (cam / (cam.max() + 1e-8)).detach().cpu().numpy()




def main():
   st = load_state()
   T = torch_parts()
   torch, device = T["torch"], T["device"]


   def vec(d, pid):
       return torch.from_numpy(d[pid] if d is not None else np.zeros(1, np.float32))[None].to(device)


   fold0_test = [p for p in st.pids if st.fold_of[p] == 0]
   chosen = [p for lab in (0, 1) for p in [q for q in fold0_test if st.labels[q] == lab][:3]]
   for name in SAVE_WEIGHTS:
       model, rad, clin = T["load_fold0"](name, st)
       if model is None or not chosen:
           print(f"Grad-CAM skipped for {name} (no fold-0 weights; train it first)"); continue
       fig, axes = plt.subplots(len(chosen), 2, figsize=(6, 2.8 * len(chosen))); axes = np.atleast_2d(axes)
       for i, pid in enumerate(chosen):
           ct, m = st.crops[pid]; j = ct.shape[0] // 2
           x = torch.from_numpy(ct[j][None]).float().to(device)
           with torch.no_grad():
               prob = torch.softmax(model(x, vec(rad, pid), vec(clin, pid)), 1)[0, 1].item()
           cam = grad_cam(T, model, x, vec(rad, pid), vec(clin, pid), model.enc.layer4, int(prob >= 0.5))
           axes[i, 0].imshow(ct[j, 0], cmap="gray", vmin=0, vmax=1); axes[i, 0].contour(m[j, 0], levels=[0.5], colors="r", linewidths=0.8)
           axes[i, 1].imshow(ct[j, 0], cmap="gray", vmin=0, vmax=1); axes[i, 1].imshow(cam, cmap="jet", alpha=0.4)
           axes[i, 1].contour(m[j, 0], levels=[0.5], colors="w", linewidths=0.6)
           axes[i, 0].set_ylabel(f"{pid}\ntrue={st.labels[pid]} p={prob:.2f}", fontsize=8)
           for a in axes[i]:
               a.set_xticks([]); a.set_yticks([])
       axes[0, 0].set_title("CT (axial) + GTV", fontsize=9); axes[0, 1].set_title("Grad-CAM (predicted class)", fontsize=9)
       plt.suptitle(f"{name}: fold-0 held-out patients"); plt.tight_layout()
       plt.savefig(f"{FIG_DIR}/explain_gradcam_{name}.png", dpi=140); plt.close()


   gate_model, g_rad, g_clin = T["load_fold0"]("gated_md", st)
   if gate_model is not None:
       rows = []
       with torch.no_grad():
           for pid in fold0_test:
               n = st.crops[pid][0].shape[0]
               gate_model(torch.from_numpy(st.crops[pid][0]).float().to(device), vec(g_rad, pid).repeat(n, 1), vec(g_clin, pid).repeat(n, 1))
               w = gate_model.last_gate.mean(0).cpu().numpy()
               rows.append({"label": st.labels[pid], "w_CT": w[0], "w_radiomics": w[1], "w_clinical": w[2]})
       G = pd.DataFrame(rows).groupby("label").mean()
       print("Mean learned modality weights by class (fold-0 held-out patients):"); print(G.round(3).to_string())
       G.plot(kind="bar", figsize=(5.5, 3.5), title="Proposed model: learned modality gate"); plt.ylabel("weight"); plt.tight_layout()
       plt.savefig(f"{FIG_DIR}/explain_modality_gate.png", dpi=140); plt.close()


   # radiomic signature: features the selection keeps when fitted on all patients (descriptive, NOT an evaluation)
   y_all = [st.labels[p] for p in st.pids]
   rp = Pipeline(rad_steps() + [("clf", LogisticRegression(C=0.5, class_weight="balanced", max_iter=5000))]).fit(
       st.xtab.loc[st.pids, st.rad_cols], y_all)
   names = [str(n) for n in   # plain str: numpy str_ mixed with str breaks sklearn feature-name checks
            np.array(st.rad_cols)[rp.named_steps["var"].get_support()][rp.named_steps["corr"].keep_][rp.named_steps["kbest"].get_support()]]
   coef = pd.Series(rp.named_steps["clf"].coef_[0], index=[n.replace("rad_", "") for n in names]).sort_values()
   plt.figure(figsize=(6.5, 4.5)); coef.plot(kind="barh", color=["#4C72B0" if v < 0 else "#DD8452" for v in coef])
   plt.title("Radiomic signature: standardised LR coefficients\n(+ = higher 2-year mortality; all patients, descriptive)")
   plt.tight_layout(); plt.savefig(f"{FIG_DIR}/explain_radiomic_signature.png", dpi=140); plt.close()
   print("Correlation of each signature feature with log tumour volume:")
   print(st.xtab.loc[st.pids, names].corrwith(st.xtab.loc[st.pids, "log_gtv_volume"]).round(2)
         .rename(lambda n: n.replace("rad_", "")).to_string())


   cols = st.clin_cols + list(names)
   Xt = pd.DataFrame(SimpleImputer(strategy="median", keep_empty_features=True).fit_transform(st.xtab.loc[st.pids, cols]), columns=cols)
   rf = RandomForestClassifier(n_estimators=500, min_samples_leaf=3, class_weight="balanced_subsample", random_state=SEED).fit(Xt, y_all)
   imp = pd.Series(permutation_importance(rf, Xt, y_all, n_repeats=20, random_state=SEED, scoring="roc_auc").importances_mean,
                   index=[c.replace("rad_", "") for c in cols]).sort_values()
   plt.figure(figsize=(6.5, 5.5)); imp.plot(kind="barh", color="#4C72B0"); plt.title("Permutation importance (RF, clinical + signature; descriptive)")
   plt.tight_layout(); plt.savefig(f"{FIG_DIR}/explain_tabular_importance.png", dpi=140); plt.close()
   try:
       import shap
       sv = shap.TreeExplainer(rf).shap_values(Xt)
       sv = sv[1] if isinstance(sv, list) else (sv[:, :, 1] if np.ndim(sv) == 3 else sv)
       shap.summary_plot(sv, Xt.rename(columns=lambda c: c.replace("rad_", "")), show=False); plt.tight_layout()
       plt.savefig(f"{FIG_DIR}/explain_tabular_shap.png", dpi=140, bbox_inches="tight"); plt.close()
   except Exception as e:
       print("SHAP summary skipped:", type(e).__name__)




if __name__ == "__main__":
   main()