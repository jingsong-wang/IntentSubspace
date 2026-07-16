#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${CISR_ENV_NAME:-CISR}"
PYTHON_VERSION="${CISR_PYTHON_VERSION:-3.11}"
RECREATE="${CISR_RECREATE:-0}"
SKIP_SMOKE="${CISR_SKIP_SMOKE:-0}"
VIDEO_EXTRAS="${CISR_VIDEO_EXTRAS:-0}"
PYTORCH_INDEX_URL="${CISR_PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"

if [[ ! -f "requirements.txt" || ! -d "src" ]]; then
  echo "Run this script from the intent_subspace repository root." >&2
  exit 1
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "conda was not found on PATH. Install Miniconda/Anaconda first, or initialize conda for this shell." >&2
  exit 1
fi

CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
  if [[ "${RECREATE}" == "1" ]]; then
    echo "Removing existing conda environment: ${ENV_NAME}"
    conda env remove -n "${ENV_NAME}" -y
  else
    echo "Conda environment '${ENV_NAME}' already exists." >&2
    echo "Set CISR_RECREATE=1 to remove and rebuild it." >&2
    exit 1
  fi
fi

echo "Creating conda environment '${ENV_NAME}' with Python ${PYTHON_VERSION}"
conda create -n "${ENV_NAME}" "python=${PYTHON_VERSION}" pip -y
conda activate "${ENV_NAME}"

python -m pip install --upgrade pip setuptools wheel packaging

echo "Installing PyTorch CUDA build from ${PYTORCH_INDEX_URL}"
python -m pip install --index-url "${PYTORCH_INDEX_URL}" torch torchvision torchaudio

echo "Installing project requirements"
python -m pip install -r requirements.txt

echo "Installing common benchmark and reproduction extras"
python -m pip install \
  "pandas>=2.2" \
  "requests>=2.32" \
  "openai>=1.78" \
  "anthropic>=0.51" \
  "sentence-transformers>=3.0" \
  "sentencepiece>=0.2" \
  "einops>=0.8" \
  "protobuf>=4.25" \
  "safetensors>=0.4" \
  "huggingface_hub>=0.31" \
  "umap-learn>=0.5" \
  "plotly>=5.20"

if [[ "${VIDEO_EXTRAS}" == "1" ]]; then
  echo "Installing optional video extras"
  python -m pip install "decord>=0.6" "av>=12"
fi

if [[ "${SKIP_SMOKE}" == "1" ]]; then
  echo "Skipping smoke tests because CISR_SKIP_SMOKE=1."
  exit 0
fi

echo "Running CUDA and package smoke test"
python - <<'PY'
import torch
import transformers
import qwen_vl_utils
import accelerate

print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
print("transformers:", transformers.__version__)
print("accelerate:", accelerate.__version__)
print("CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available inside the CISR environment.")
for idx in range(torch.cuda.device_count()):
    print(f"cuda:{idx}", torch.cuda.get_device_name(idx), torch.cuda.get_device_capability(idx))
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration  # noqa: F401
print("Qwen2.5-VL transformers class: OK")
print("qwen-vl-utils:", getattr(qwen_vl_utils, "__version__", "installed"))
PY

echo "Running project mock smoke test"
python -m jailbreak_repro.run_experiment \
  --model mock \
  --model-backend mock \
  --attack figstep \
  --defense ecso \
  --dataset tiny \
  --max-samples 1 \
  --out-dir runs/_env_smoke_cisr_mock

echo "CISR environment is ready. Activate it with: conda activate ${ENV_NAME}"
