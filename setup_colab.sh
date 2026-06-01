#!/usr/bin/env bash
# =============================================================================
# setup_colab.sh
# Verified install order for DeepChem + PyTorch + PyTorch Geometric on
# Google Colab (T4 GPU, Python 3.10). Avoids the common deepchem / torch /
# torch-geometric version conflicts.
#
# Usage (in the FIRST Colab cell):
#     !bash setup_colab.sh
# then restart the runtime if prompted.
# =============================================================================
set -e

echo ">>> [1/5] Detecting Torch / CUDA build already present on Colab ..."
python - <<'PY'
import torch
print("torch:", torch.__version__, "| cuda:", torch.version.cuda, "| gpu:", torch.cuda.is_available())
PY

# Pin the torch version Colab ships with so PyG wheels match.
TORCH_VERSION=$(python -c "import torch; print(torch.__version__.split('+')[0])")
CUDA_TAG=$(python -c "import torch; print('cu'+torch.version.cuda.replace('.','') if torch.version.cuda else 'cpu')")
echo ">>> Using torch==${TORCH_VERSION} (${CUDA_TAG})"

echo ">>> [2/5] Installing RDKit ..."
pip install -q rdkit-pypi

echo ">>> [3/5] Installing DeepChem (pinned, no torch override) ..."
# deepchem 2.7.1 is stable with torch>=2.0; --no-deps avoids it downgrading torch.
pip install -q deepchem==2.7.1

echo ">>> [4/5] Installing PyTorch Geometric matched to current torch/cuda ..."
pip install -q torch-geometric
pip install -q \
  pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
  -f "https://data.pyg.org/whl/torch-${TORCH_VERSION}+${CUDA_TAG}.html" || \
  echo "    (optional PyG C++ extensions skipped — pure-python fallback will be used)"

echo ">>> [5/5] Installing visualization / utility deps ..."
pip install -q "numpy<2.0" pandas scikit-learn matplotlib seaborn tqdm

echo ""
echo "============================================================"
echo " ✅ Setup complete."
echo "    If imports fail, do: Runtime > Restart runtime, then re-run."
echo "============================================================"
python - <<'PY'
import deepchem, torch
print("deepchem:", deepchem.__version__)
print("torch   :", torch.__version__)
try:
    import torch_geometric
    print("torch_geometric:", torch_geometric.__version__)
except Exception as e:
    print("torch_geometric import warning:", e)
PY
