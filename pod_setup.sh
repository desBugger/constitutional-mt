#!/usr/bin/env bash
# pod_setup.sh — Run once after deploying a fresh RunPod 4×A100 pod.
# Fixes the vLLM/transformers compatibility stack for Nemotron-Super (nemotron_h).
#
# Known working combination (July 2026):
#   RunPod driver: 570.172.08  (CUDA 12.8 max)
#   vLLM:          0.17.1      (cu128, NemotronH support added)
#   torch:         2.10.0+cu128
#   transformers:  5.13.0      (knows nemotron_h config class)
#   Patches:       nemotron_h valid_types + reverse_mapping for legacy mamba/attention names
#
# Usage:  HF_TOKEN=hf_... bash pod_setup.sh
set -euo pipefail

echo "=== [1/5] Installing vLLM 0.17.1 (cu128) ==="
pip install \
  "$(ls /tmp/vllm_dl17/vllm-0.17.1-*.whl 2>/dev/null || echo 'vllm==0.17.1')" \
  --break-system-packages -q 2>&1 | tail -3

echo "=== [2/5] Pinning torch to 2.10.0+cu128 ==="
pip install \
  "torch==2.10.0+cu128" \
  "torchaudio==2.10.0+cu128" \
  "torchvision==0.25.0+cu128" \
  --index-url https://download.pytorch.org/whl/cu128 \
  --break-system-packages --force-reinstall -q 2>&1 | tail -3

echo "=== [3/5] Upgrading transformers to 5.13.0 ==="
pip install "transformers==5.13.0" --break-system-packages -q 2>&1 | tail -3

echo "=== [4/5] Patching transformers nemotron_h config (legacy layer names) ==="
python3 << 'PYEOF'
import sys

fpath = '/usr/local/lib/python3.12/dist-packages/transformers/models/nemotron_h/configuration_nemotron_h.py'
with open(fpath) as f:
    content = f.read()

patches = [
    (
        '        valid_types = {"full_attention", "linear_attention", "moe", "mlp"}',
        '        valid_types = {"full_attention", "linear_attention", "moe", "mlp", "mamba", "attention"}',
    ),
    (
        '        reverse_mapping = {"linear_attention": "M", "moe": "E", "full_attention": "*", "mlp": "-"}',
        '        reverse_mapping = {"linear_attention": "M", "moe": "E", "full_attention": "*", "mlp": "-", "mamba": "M", "attention": "*"}',
    ),
]

changed = 0
for old, new in patches:
    if old in content:
        content = content.replace(old, new)
        changed += 1
        print(f"  patched: {old[:60]}...")
    elif new in content:
        print(f"  already patched: {new[:60]}...")
        changed += 1
    else:
        print(f"  ERROR: pattern not found: {old[:60]}", file=sys.stderr)
        sys.exit(1)

with open(fpath, 'w') as f:
    f.write(content)

print(f"  {changed}/2 patches applied to {fpath}")
PYEOF

echo "=== [5/5] Verifying CUDA access ==="
python3 -c "
import torch
print(f'  torch:          {torch.__version__}')
print(f'  cuda available: {torch.cuda.is_available()}')
print(f'  gpu count:      {torch.cuda.device_count()}')
assert torch.cuda.is_available(), 'CUDA not available!'
"

echo "=== [6/6] Downloading MASK eval data (gated — needs HF_TOKEN) ==="
HF_TOKEN="${HF_TOKEN:-}"
if [ -z "$HF_TOKEN" ]; then
    echo "  WARNING: HF_TOKEN not set — skipping MASK download."
    echo "  Run manually: HF_TOKEN=<token> python3 evals/sample_mask.py --download --hf-token \$HF_TOKEN"
else
    python3 evals/sample_mask.py --download --hf-token "$HF_TOKEN"
fi

echo ""
echo "=== Setup complete. ==="
echo "Start vLLM with:"
echo "  FLASHINFER_DISABLE_VERSION_CHECK=1 python3 -m vllm.entrypoints.openai.api_server \\"
echo "    --model /root/models/<checkpoint_id> \\"
echo "    --served-model-name <checkpoint_id> \\"
echo "    --tensor-parallel-size 4 --trust-remote-code --port 8000 --dtype bfloat16"
