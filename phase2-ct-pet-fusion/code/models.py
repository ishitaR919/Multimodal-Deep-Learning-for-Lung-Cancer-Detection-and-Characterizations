"""Phase 2 networks, slice dataset, training / prediction and Grad-CAM (torch).


All models take x = (B, 2, H, W) with channel 0 = CT, channel 1 = PET, and an optional clinical vector.
Crops and labels are passed in explicitly (see common.load_state()).
"""
import os
import random


import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import Dataset, DataLoader


from config import SEED, EPOCHS, BATCH_SIZE, LR, CROP_PX, MODEL_SPECS, SAVE_WEIGHTS, WEIGHT_DIR
from common import roc_auc_safe


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")




def seed_everything(s):
   random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def prep(t):                      # inputs are in [0, 1]
   return (t - 0.5) / 0.25


def gap(f):
   return F.adaptive_avg_pool2d(f, 1).flatten(1)


class Encoder(nn.Module):
   """ResNet-18 trunk returning the layer4 feature map (B, 512, H/32, W/32)."""
   def __init__(self, in_ch=1, pretrained=True):
       super().__init__()
       try:
           base = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
       except Exception as e:
           print("Pretrained weights unavailable (is Internet on?) - using random init:", type(e).__name__)
           base = torchvision.models.resnet18(weights=None)
       if in_ch != 3:
           old = base.conv1.weight.data
           conv = nn.Conv2d(in_ch, 64, 7, 2, 3, bias=False)
           conv.weight.data = old.mean(dim=1, keepdim=True).repeat(1, in_ch, 1, 1) * (3.0 / in_ch)
           base.conv1 = conv
       self.stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
       self.layer1, self.layer2, self.layer3, self.layer4 = base.layer1, base.layer2, base.layer3, base.layer4


   def forward(self, x):
       return self.layer4(self.layer3(self.layer2(self.layer1(self.stem(x)))))


class UniNet(nn.Module):
   def __init__(self, channel):
       super().__init__()
       self.channel, self.enc = channel, Encoder(1)
       self.drop, self.fc = nn.Dropout(0.3), nn.Linear(512, 2)


   def forward(self, x, clin=None):
       c = self.channel
       return self.fc(self.drop(gap(self.enc(prep(x[:, c:c + 1])))))


class EarlyNet(nn.Module):
   """Early fusion: CT and PET stacked as a 2-channel image."""
   def __init__(self):
       super().__init__()
       self.enc, self.drop, self.fc = Encoder(2), nn.Dropout(0.3), nn.Linear(512, 2)


   def forward(self, x, clin=None):
       return self.fc(self.drop(gap(self.enc(prep(x)))))


class MidNet(nn.Module):
   """Intermediate (feature-level) fusion: separate encoders, concatenated embeddings (+ optional clinical)."""
   def __init__(self, n_clin=0):
       super().__init__()
       self.enc_ct, self.enc_pet, self.n_clin = Encoder(1), Encoder(1), n_clin
       if n_clin:
           self.clin_mlp = nn.Sequential(nn.Linear(n_clin, 32), nn.ReLU(), nn.Dropout(0.2))
       self.head = nn.Sequential(nn.Dropout(0.4), nn.Linear(1024 + (32 if n_clin else 0), 256), nn.ReLU(),
                                 nn.Dropout(0.3), nn.Linear(256, 2))


   def forward(self, x, clin=None):
       feats = [gap(self.enc_ct(prep(x[:, 0:1]))), gap(self.enc_pet(prep(x[:, 1:2])))]
       if self.n_clin:
           feats.append(self.clin_mlp(clin))
       return self.head(torch.cat(feats, 1))


class CrossAttnNet(nn.Module):
   """Cross-modal attention fusion: CT tokens attend to PET tokens and vice versa, then a learned
   modality gate weights the two pooled streams (+ optional clinical branch)."""
   def __init__(self, n_clin=0, d=128, heads=4):
       super().__init__()
       n_tok = (CROP_PX // 32) ** 2
       self.enc_ct, self.enc_pet, self.n_clin = Encoder(1), Encoder(1), n_clin
       self.proj_ct, self.proj_pet = nn.Conv2d(512, d, 1), nn.Conv2d(512, d, 1)
       self.pos_ct, self.pos_pet = nn.Parameter(torch.zeros(1, n_tok, d)), nn.Parameter(torch.zeros(1, n_tok, d))
       nn.init.trunc_normal_(self.pos_ct, std=0.02); nn.init.trunc_normal_(self.pos_pet, std=0.02)
       self.attn_ct = nn.MultiheadAttention(d, heads, batch_first=True, dropout=0.1)
       self.attn_pet = nn.MultiheadAttention(d, heads, batch_first=True, dropout=0.1)
       self.norm_ct, self.norm_pet = nn.LayerNorm(d), nn.LayerNorm(d)
       self.gate = nn.Sequential(nn.Linear(2 * d, 32), nn.ReLU(), nn.Linear(32, 2))
       if n_clin:
           self.clin_mlp = nn.Sequential(nn.Linear(n_clin, 32), nn.ReLU(), nn.Dropout(0.2))
       self.head = nn.Sequential(nn.Dropout(0.4), nn.Linear(2 * d + (32 if n_clin else 0), 128), nn.ReLU(),
                                 nn.Dropout(0.3), nn.Linear(128, 2))
       self.last_gate = None


   def forward(self, x, clin=None):
       tc = self.proj_ct(self.enc_ct(prep(x[:, 0:1]))).flatten(2).transpose(1, 2) + self.pos_ct     # (B, T, d)
       tp = self.proj_pet(self.enc_pet(prep(x[:, 1:2]))).flatten(2).transpose(1, 2) + self.pos_pet
       ac, _ = self.attn_ct(tc, tp, tp)
       ap, _ = self.attn_pet(tp, tc, tc)
       pc, pp = self.norm_ct(tc + ac).mean(1), self.norm_pet(tp + ap).mean(1)
       w = torch.softmax(self.gate(torch.cat([pc, pp], 1)), dim=1)                                    # modality weights
       self.last_gate = w.detach()
       feats = [w[:, 0:1] * pc, w[:, 1:2] * pp]
       if self.n_clin:
           feats.append(self.clin_mlp(clin))
       return self.head(torch.cat(feats, 1))


def build_model(name, n_clin=0):
   s = MODEL_SPECS[name]
   if s["kind"] == "uni":
       return UniNet(s["channel"])
   if s["kind"] == "early":
       return EarlyNet()
   if s["kind"] == "mid":
       return MidNet(n_clin if s["clin"] else 0)
   return CrossAttnNet(n_clin if s["clin"] else 0)


def modality_dropout(x, p):
   """Per sample: zero the CT channel with prob p, or (exclusively) the PET channel with prob p."""
   r = torch.rand(x.shape[0], device=x.device)
   x = x.clone()
   x[r < p, 0] = 0.0
   x[(r >= p) & (r < 2 * p), 1] = 0.0
   return x


class SliceDataset(Dataset):
   def __init__(self, crops, pids, labels, clin=None, augment=False):
       self.crops, self.pids, self.labels, self.clin, self.augment = crops, list(pids), labels, clin, augment
       self.index = [(pi, j) for pi, pid in enumerate(self.pids) for j in range(crops[pid][0].shape[0])]


   def __len__(self):
       return len(self.index)


   def __getitem__(self, i):
       pi, j = self.index[i]
       pid = self.pids[pi]
       ct, pet = self.crops[pid]
       x = np.stack([ct[j], pet[j]]).astype(np.float32)
       if self.augment:                                   # identical spatial transform for CT and PET
           if random.random() < 0.5:
               x = x[:, :, ::-1]
           if random.random() < 0.5:
               x = x[:, ::-1, :]
           x = np.roll(x, (random.randint(-8, 8), random.randint(-8, 8)), axis=(1, 2))
       clin = self.clin[pid] if self.clin is not None else np.zeros(1, np.float32)
       return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(clin), int(self.labels[pid]), pi


def make_loader(ds, train):
   return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=train, num_workers=0, drop_last=(train and len(ds) > BATCH_SIZE))


def predict(model, crops, pids, labels, clin=None, missing=None):
   """Patient-level probability = mean over that patient's slices. missing in {None,'ct','pet'}."""
   ds = SliceDataset(crops, pids, labels, clin, augment=False)
   model.eval()
   probs, pidx, loss_sum = [], [], 0.0
   for x, c, y, pi in make_loader(ds, False):
       x, c = x.to(DEVICE), c.to(DEVICE)
       if missing == "ct":
           x = x.clone(); x[:, 0] = 0.0
       elif missing == "pet":
           x = x.clone(); x[:, 1] = 0.0
       logits = model(x, c)
       loss_sum += F.cross_entropy(logits, y.to(DEVICE), reduction="sum").item()
       probs.append(torch.softmax(logits, 1)[:, 1].cpu().numpy()); pidx.append(pi.numpy())
   g = pd.DataFrame({"pi": np.concatenate(pidx), "p": np.concatenate(probs)}).groupby("pi")["p"].mean()
   out = pd.DataFrame({"patient_id": [ds.pids[i] for i in g.index], "prob": g.values})
   out["label"] = out["patient_id"].map(labels).astype(int)
   return out, loss_sum / max(len(ds), 1)


def train_fold(name, fold, tr, va, te, crops, labels, clin_dict, n_clin, epochs=None):
   epochs = epochs or EPOCHS
   seed_everything(SEED + fold)
   spec = MODEL_SPECS[name]
   model = build_model(name, n_clin).to(DEVICE)
   cl = clin_dict if spec["clin"] else None
   cnt = np.bincount([labels[p] for p in tr], minlength=2)
   w = torch.tensor(len(tr) / (2.0 * np.maximum(cnt, 1)), dtype=torch.float32, device=DEVICE)
   crit = nn.CrossEntropyLoss(weight=w)
   opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
   sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
   loader = make_loader(SliceDataset(crops, tr, labels, cl, augment=True), True)


   hist, best_loss, best_state = [], float("inf"), None
   for ep in range(epochs):
       model.train()
       run, n = 0.0, 0
       for x, c, y, _ in loader:
           x, c, y = x.to(DEVICE), c.to(DEVICE), y.to(DEVICE)
           if spec["md"] > 0:
               x = modality_dropout(x, spec["md"])
           opt.zero_grad()
           loss = crit(model(x, c), y)
           loss.backward()
           torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
           opt.step()
           run += loss.item() * len(y); n += len(y)
       sched.step()
       vdf, vloss = predict(model, crops, va, labels, cl)
       hist.append({"epoch": ep + 1, "train_loss": run / max(n, 1), "val_loss": vloss,
                    "val_auc": roc_auc_safe(vdf["label"], vdf["prob"]),
                    "val_acc": float(((vdf["prob"] >= 0.5).astype(int) == vdf["label"]).mean())})
       if vloss < best_loss:                               # model selection on validation loss (stabler than AUC on ~15 patients)
           best_loss, best_state = vloss, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
   model.load_state_dict(best_state)
   test_df, _ = predict(model, crops, te, labels, cl)
   test_df["fold"] = fold
   miss = {}
   if spec["both"]:                                        # same trained model, one modality blanked at test time
       for mode in ("ct", "pet"):
           miss[mode] = predict(model, crops, te, labels, cl, missing=mode)[0].assign(fold=fold)
   if fold == 0 and name in SAVE_WEIGHTS:
       torch.save(best_state, f"{WEIGHT_DIR}/{name}_fold0.pt")
   return test_df, pd.DataFrame(hist).assign(model=name, fold=fold), miss


def grad_cam(model, x, clin, layer, class_idx):
   store = {}


   def fwd_hook(module, inp, out):
       store["a"] = out
       out.register_hook(lambda g: store.__setitem__("g", g))


   handle = layer.register_forward_hook(fwd_hook)
   model.zero_grad()
   with torch.enable_grad():
       out = model(x, clin)
       out[0, class_idx].backward()
   handle.remove()
   cam = F.relu((store["g"].mean(dim=(2, 3), keepdim=True) * store["a"]).sum(1, keepdim=True))
   cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
   cam = cam - cam.min()
   return (cam / (cam.max() + 1e-8)).detach().cpu().numpy()


def load_fold0(name):
   p = f"{WEIGHT_DIR}/{name}_fold0.pt"
   if not os.path.exists(p):
       return None
   model = build_model(name, 0).to(DEVICE)
   model.load_state_dict(torch.load(p, map_location=DEVICE)); model.eval()
   return model