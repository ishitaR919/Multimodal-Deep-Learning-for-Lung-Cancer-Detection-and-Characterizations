"""Step 6 - Grad-CAM heat maps for 3 correct and 3 wrong test patches per trained model.


Output (FIG_ROOT/gradcam): <model>_gradcam.png
"""
import os


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


from config import PREP_DIR, OUT_DIR, FIG_ROOT, MODELS, INPUT_SIZE
from common import DEVICE, to_model_input, build_model


FIG_DIR = f"{FIG_ROOT}/gradcam"


# last convolutional block of each architecture (Grad-CAM needs its feature maps)
TARGET_LAYERS = {
   "resnet50": lambda m: [m.layer4[-1]],
   "densenet121": lambda m: [m.features[-1]],
   "efficientnet_b0": lambda m: [m.features[-1]],
   "vgg16_bn": lambda m: [m.features[-1]],
}




def load_trained_model(model_name):
   """Rebuild the architecture and load its best checkpoint."""
   model = build_model(model_name)
   model.load_state_dict(torch.load(f"{OUT_DIR}/{model_name}_best.pt", map_location=DEVICE))
   model.eval()
   return model




def run_gradcam_for_model(model_name, n_correct=3, n_incorrect=3):
   preds_csv = f"{OUT_DIR}/{model_name}_test_predictions.csv"
   if not os.path.exists(preds_csv):
       print(f"No results found for {model_name}, skipping.")
       return


   model = load_trained_model(model_name)
   cam = GradCAM(model=model, target_layers=TARGET_LAYERS[model_name](model))


   preds_df = pd.read_csv(preds_csv)
   correct_df = preds_df[preds_df["label"] == preds_df["pred"]]
   incorrect_df = preds_df[preds_df["label"] != preds_df["pred"]]
   combined = pd.concat([
       correct_df.sample(min(n_correct, len(correct_df)), random_state=42).assign(status="correct"),
       incorrect_df.sample(min(n_incorrect, len(incorrect_df)), random_state=42).assign(status="incorrect"),
   ])
   n = len(combined)
   if n == 0:
       print(f"No samples to visualize for {model_name}.")
       return


   fig, axes = plt.subplots(2, n, figsize=(4 * n, 8))
   if n == 1:
       axes = axes.reshape(2, 1)


   for col_idx, (_, row) in enumerate(combined.iterrows()):
       crop = torch.from_numpy(np.load(os.path.join(PREP_DIR, row["filename"]))).float()  # (3, H, W) in [0,1]
       input_tensor = to_model_input(crop).unsqueeze(0).to(DEVICE)
       axial = transforms.functional.resize(crop[0:1], [INPUT_SIZE, INPUT_SIZE], antialias=True)[0]
       axial = axial.clamp(0, 1).numpy()


       grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(int(row["pred"]))])[0]
       rgb_img = np.stack([axial, axial, axial], axis=-1).astype(np.float32)  # HxWx3 for overlay
       cam_overlay = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)


       axes[0, col_idx].imshow(axial, cmap="gray")
       axes[0, col_idx].set_title(f"Original\ntrue={row['label']} pred={row['pred']} ({row['status']})", fontsize=9)
       axes[0, col_idx].axis("off")
       axes[1, col_idx].imshow(cam_overlay)
       axes[1, col_idx].set_title("Grad-CAM overlay (axial)", fontsize=9)
       axes[1, col_idx].axis("off")


   plt.suptitle(f"Grad-CAM explanations - {model_name}", fontsize=13)
   plt.tight_layout()
   save_path = f"{FIG_DIR}/{model_name}_gradcam.png"
   plt.savefig(save_path, dpi=150)
   plt.close()
   print(f"Saved {save_path}")




def main():
   os.makedirs(FIG_DIR, exist_ok=True)
   trained_models = [name for name in MODELS if os.path.exists(f"{OUT_DIR}/{name}_best.pt")]
   print(f"Running Grad-CAM for: {trained_models}")
   for name in trained_models:
       run_gradcam_for_model(name)




if __name__ == "__main__":
   main()
