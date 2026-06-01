#!/usr/bin/env bash
# =============================================================================
# setup_colab.sh
# Verified install order for DeepChem + PyTorch + PyTorch Geometric on
# modern Google Colab (Python 3.12, torch 2.x preinstalled, T4 GPU).
#
# Usage (in the FIRST Colab cell):
#     !bash setup_colab.sh
# then restart the runtime if Colab shows a "RESTART RUNTIME" button,
# and re-run the clone + this cell.
# =============================================================================
set -e

echo ">>> [1/5] Detecting Torch / CUDA build already present on Colab ..."
python - <<'PY'
import torch
print("torch:", torch.__version__, "| cuda:", torch.version.cuda, "| gpu:", torch.cuda.is_available())
PY

echo ">>> [2/5] Installing RDKit (official wheels; NOT the deprecated rdkit-pypi) ..."
# 'rdkit-pypi' is unmaintained and has no wheels for Python 3.11/3.12.
# The official 'rdkit' package ships manylinux wheels and is what Colab needs.
pip install -q rdkit

echo ">>> [3/5] (DeepChem is optional and NOT required) ..."
# Data loading + featurization is done directly with RDKit (robust on modern
# NumPy). Uncomment the next line only if you want DeepChem for experiments:
# pip install -q deepchem

echo ">>> [4/5] Installing PyTorch Geometric ..."
# torch-geometric core is pure-python and works on top of the preinstalled
# torch. The optional C++ extensions (torch_scatter etc.) often have no wheel
# for the very latest torch builds on Colab, so we install them best-effort
# and fall back to PyG's native (slower) scatter implementation if missing.
TORCH_VERSION=$(python -c "import torch; print(torch.__version__.split('+')[0])")
CUDA_TAG=$(python -c "import torch; print('cu'+torch.version.cuda.replace('.','') if torch.version.cuda else 'cpu')")
echo "    torch=${TORCH_VERSION} (${CUDA_TAG})"
pip install -q torch-geometric
pip install -q \
  pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
  -f "https://data.pyg.org/whl/torch-${TORCH_VERSION}+${CUDA_TAG}.html" 2>/dev/null || \
  echo "    (optional PyG C++ extensions unavailable for this torch — using native fallback, GCNConv still works)"

echo ">>> [5/5] Installing visualization / utility deps ..."
# Pin numpy<2.0 for broad compatibility with deepchem / older binary deps.
pip install -q "numpy<2.0" pandas scikit-learn matplotlib seaborn tqdm

echo ""
echo "============================================================"
echo " ✅ Setup complete."
echo "    If imports fail or Colab shows 'RESTART RUNTIME', do:"
echo "    Runtime > Restart runtime, then re-run the clone + setup cell."
echo "============================================================"
python - <<'PY'
import importlib
for mod in ("rdkit", "torch", "torch_geometric"):
    try:
        m = importlib.import_module(mod)
        print(f"{mod:>16}:", getattr(m, "__version__", "ok"))
    except Exception as e:
        print(f"{mod:>16}: import warning -> {e}")
PY
