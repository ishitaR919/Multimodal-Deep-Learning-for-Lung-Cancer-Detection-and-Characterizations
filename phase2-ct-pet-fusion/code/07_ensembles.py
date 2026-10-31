"""Step 7 - late fusion and voting ensembles, built from the saved out-of-fold patient predictions.
Members are fixed in advance (no 'pick the best k on the same OOF data') to avoid selection bias.


Usage:  python 07_ensembles.py


Input:  oof_<model>.csv (steps 5-6)
Output (RES_DIR): oof_late_ct_pet.csv, oof_late_ct_pet_clin.csv, oof_ens_{soft,hard}_fusion.csv,
                 oof_ens_{soft,hard}_image_all.csv
"""
import pandas as pd


from config import RES_DIR, MODEL_SPECS, TAB_MODELS
from common import load_oof




def soft_combine(oof, members):
   """Late fusion / soft voting: average the members' patient-level probabilities."""
   P = pd.concat([oof[m].set_index("patient_id")["prob"].rename(m) for m in members], axis=1)
   base = oof[members[0]].set_index("patient_id")[["label", "fold"]]
   return base.assign(prob=P.mean(axis=1)).reset_index()




def hard_combine(oof, members):
   """Hard (majority) voting: each member votes at 0.5; prob = fraction of positive votes (ties count as positive)."""
   P = pd.concat([oof[m].set_index("patient_id")["prob"].rename(m) for m in members], axis=1)
   base = oof[members[0]].set_index("patient_id")[["label", "fold"]]
   return base.assign(prob=(P >= 0.5).mean(axis=1)).reset_index()




COMBOS = {
   "late_ct_pet": (soft_combine, ["ct_only", "pet_only"]),
   "late_ct_pet_clin": (soft_combine, ["ct_only", "pet_only", "tab_clinical_lr"]),
   "ens_soft_fusion": (soft_combine, ["early", "intermediate", "cross_attn_md"]),
   "ens_hard_fusion": (hard_combine, ["early", "intermediate", "cross_attn_md"]),
   "ens_soft_image_all": (soft_combine, ["ct_only", "pet_only", "early", "intermediate", "cross_attn", "cross_attn_md"]),
   "ens_hard_image_all": (hard_combine, ["ct_only", "pet_only", "early", "intermediate", "cross_attn", "cross_attn_md"]),
}




def main():
   oof = load_oof(TAB_MODELS + list(MODEL_SPECS))
   for cname, (fn, members) in COMBOS.items():
       mem = [m for m in members if m in oof]
       if len(mem) >= 2:
           fn(oof, mem).to_csv(f"{RES_DIR}/oof_{cname}.csv", index=False)
           print(f"{cname:22s} <- {mem}")
       else:
           print(f"{cname:22s} skipped (needs >=2 trained members)")




if __name__ == "__main__":
   main()