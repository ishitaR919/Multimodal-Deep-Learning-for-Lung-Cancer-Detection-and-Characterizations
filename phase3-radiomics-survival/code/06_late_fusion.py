"""Step 6 - late fusion: average the out-of-fold probabilities of separately trained models.


Members are fixed in advance (no 'pick the best k on the same out-of-fold data'), so there is no selection bias.


Usage:   python 06_late_fusion.py
Inputs:  RES_DIR/oof_ct_cnn.csv, oof_tab_radiomics_lr.csv, oof_tab_clinical_lr.csv
Outputs: RES_DIR/oof_late_ct_rad.csv, RES_DIR/oof_late_ct_rad_clin.csv
"""
import pandas as pd


from config import RES_DIR
from common import load_oof


COMBOS = {"late_ct_rad": ["ct_cnn", "tab_radiomics_lr"],
         "late_ct_rad_clin": ["ct_cnn", "tab_radiomics_lr", "tab_clinical_lr"]}




def soft_combine(oof, members):
   P = pd.concat([oof[m].set_index("patient_id")["prob"].rename(m) for m in members], axis=1)
   base = oof[members[0]].set_index("patient_id")[["label", "fold"]]
   return base.assign(prob=P.mean(axis=1)).reset_index()




def main():
   oof = load_oof({m for ms in COMBOS.values() for m in ms})
   for cname, members in COMBOS.items():
       mem = [m for m in members if m in oof]
       if len(mem) >= 2:
           soft_combine(oof, mem).to_csv(f"{RES_DIR}/oof_{cname}.csv", index=False)
           print(f"{cname:18s} <- {mem}")
       else:
           print(f"{cname:18s} skipped (needs >= 2 trained members)")




if __name__ == "__main__":
   main()
