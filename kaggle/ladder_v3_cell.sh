# Run in a Kaggle %%bash cell with Internet and a T4 enabled.
set -euo pipefail
: "${LADDER_V3_COMMIT:?HARD STOP: set the full registered Git commit SHA}"
[[ "$LADDER_V3_COMMIT" =~ ^[0-9a-f]{40}$ ]] || { echo 'HARD STOP: invalid commit'; exit 1; }
out=/kaggle/working/ladder-v3
[[ ! -e "$out" ]] || { echo "HARD STOP: $out already exists"; exit 1; }
mkdir "$out"
export TMPDIR="$out/tmp" HF_HOME="$out/hf" XDG_CACHE_HOME="$out/cache"
export MPLCONFIGDIR="$out/matplotlib" PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1
mkdir "$TMPDIR"
git clone --no-checkout https://github.com/sohamsssssssssssssssss/sih2026.git "$out/repo"
git -C "$out/repo" checkout --detach "$LADDER_V3_COMMIT"
python -m venv --system-site-packages "$out/venv"
"$out/venv/bin/python" -m pip install --no-cache-dir 'transformers==4.49.0' 'accelerate==1.5.2' 'peft==0.15.2' 'qwen-vl-utils==0.0.8'
cd "$out/repo"
export PYTHONPATH="$out/repo"
"$out/venv/bin/python" "$out/repo/kaggle/run_ladder_v3.py" --commit "$LADDER_V3_COMMIT"
