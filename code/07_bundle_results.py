"""Step 7 - zip figures/ and model_results/ (without .pt checkpoints) into phase1_results_bundle.zip."""
import os
import shutil
from config import WORK_DIR, OUT_DIR, FIG_ROOT


def main():
   stage = f"{WORK_DIR}/_zip_stage"
   shutil.rmtree(stage, ignore_errors=True)
   shutil.copytree(FIG_ROOT, f"{stage}/figures")
   shutil.copytree(OUT_DIR, f"{stage}/model_results", ignore=shutil.ignore_patterns("*.pt"))
   zp = shutil.make_archive(f"{WORK_DIR}/phase1_results_bundle", "zip", stage)
   shutil.rmtree(stage)
   print(f"Created {zp} ({os.path.getsize(zp) / 1e6:.1f} MB)")


if __name__ == "__main__":
   main()
