"""Run this first:  python check_setup.py
Checks the Python packages, whether a GPU is usable, and runs one short training pass."""
import importlib, os, shutil, subprocess, sys, time
ok = True
for pkg in ["torch", "numpy", "pandas", "sklearn", "matplotlib", "openpyxl"]:
    try:
        m = importlib.import_module(pkg); print(f"  {pkg:<11} {getattr(m, '__version__', '')}")
    except ImportError:
        ok = False; print(f"  {pkg:<11} MISSING  ->  pip install -r requirements.txt")
if not ok:
    sys.exit(1)
import torch
print(f"  CPU threads {torch.get_num_threads()}")
if torch.cuda.is_available():
    print(f"  GPU         {torch.cuda.get_device_name(0)}  (training will use it)")
else:
    print("  GPU         none usable -> training runs on the CPU")
here = os.path.dirname(os.path.abspath(__file__))
if not os.path.exists(os.path.join(here, "output", "tensors_v3", "tensors.npz")):
    sys.exit("  tensors missing: output/tensors_v3/tensors.npz")
print("Short test (1 epoch, fold 1, tuning mode, nothing is kept)...")
t = time.time()
r = subprocess.run([sys.executable, os.path.join(here, "scripts", "train_cv.py"), "--model", "slopecnn",
                    "--channels", "all6", "--pore_pressure", "--tune", "--lr", "1e-3", "--folds", "1",
                    "--seeds", "0", "--epochs", "1", "--out", os.path.join(here, "output", "_check")])
sec = time.time() - t
print(f"  one SlopeCNN epoch: about {sec:.0f} s"
      f"  -> run_paper roughly {sec * 87 * 60 * 1.5 / 3600:.0f} h (87 (fold, seed) runs x ~60 epochs, Soranzo ~2x slower)"
      if r.returncode == 0 else "  test FAILED - see the error above")
shutil.rmtree(os.path.join(here, "output", "_check"), ignore_errors=True)
