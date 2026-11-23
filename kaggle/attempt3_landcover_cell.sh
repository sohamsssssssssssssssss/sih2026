set -euo pipefail
trap 'echo "FAILED at line $LINENO: $BASH_COMMAND" >&2' ERR
stop() { echo "HARD STOP: $*" >&2; exit 1; }
W=/kaggle/working/ben-attempt3
[[ ! -e "$W" ]] || stop "$W already exists"

one() {
  local kind="$1"
  shift
  local -a found
  mapfile -t found < <(find /kaggle/input -type f "$@")
  [[ ${#found[@]} == 1 ]] || stop "expected one $kind, found ${#found[@]}"
  echo "${found[0]}"
}

check() {
  [[ -f "$2" && "$(sha256sum "$2" | cut -d' ' -f1)" == "$1" ]] || stop "hash mismatch or missing file: $2"
}

mapfile -t code_zips < <(find /kaggle/input -type f -name satquery-ben-attempt3-code.zip)
mapfile -t new_zips < <(find /kaggle/input -type f -name satquery-ben-attempt3-worldcover.zip)
[[ ${#code_zips[@]} -le 1 && ${#new_zips[@]} -le 1 ]] || stop "ambiguous attempt-3 artifacts"
old_manifest=$(one old-worldcover -path '*/ben-wc-transfer-pack-zip/*' -name MANIFEST.json)
s1_checkpoint=$(one Europe-S1 -path '*/ben-transfer-run-1/*' -name ben-landcover-s1-checkpoint.pt)
s2_checkpoint=$(one Europe-S2 -path '*/ben-transfer-run-1/*' -name ben-landcover-s2-checkpoint.pt)
[[ "$(dirname "$s1_checkpoint")" == "$(dirname "$s2_checkpoint")" ]] || stop "checkpoint paths disagree"
check 5cc82c713ad76a3286afd4ad3eb66f33da3fb7ff365ab23a6371dc3ba999c40d "$old_manifest"
check 763c5450da2edc08a2fe2374de7cfcf810f4d5befdff330d0ca0106f3f630bb5 "$s1_checkpoint"
check 703824bb60e4d61477897f94969bf1abdbbf7621a6149a7f7b56e2c4eb6c85fa "$s2_checkpoint"
nvidia-smi --query-gpu=name,memory.total --format=csv
python -c 'import torch; assert torch.cuda.is_available(), "HARD STOP: CUDA is required"'
mkdir -p "$W"/{code,new,tmp}
if [[ ${#code_zips[@]} == 1 ]]; then
  check 7025d61c59e2aa87adfcddf59c4cb0e584e76430dff88a0edc85b44756ed245c "${code_zips[0]}"
  unzip -q "${code_zips[0]}" -d "$W/code"
  code="$W/code"
else
  sums=$(one extracted-code -name ATTEMPT3-CODE-SHA256SUMS)
  code=$(dirname "$sums")
fi
if [[ ${#new_zips[@]} == 1 ]]; then
  check 889c93aa64301394c885062fda71cba386435f811cae4251a1340406430b7b95 "${new_zips[0]}"
  unzip -q "${new_zips[0]}" -d "$W/new"
  new="$W/new"
else
  manifest=$(one extracted-new-worldcover -path '*/satquery-ben-attempt3-worldcover/*' -name MANIFEST.json)
  new=$(dirname "$manifest")
fi
check b28e1171ddd5012eaa5a180d05763f8785e00efc9295483d56bf69166a5c266b "$code/ATTEMPT3-CODE-SHA256SUMS"
(cd "$code" && sha256sum --strict -c ATTEMPT3-CODE-SHA256SUMS)
check 6a7525fafdf0266951c1139a26e8346461da678dcb5bf2844cc646fc4ab13492 "$new/MANIFEST.json"
check d14b345b6f4cabae4d713864244ea8f55d5fd0f08e1f7bb272c8240009ec6f8f "$code/group-split.json"

export PYTHONPATH="$code" PYTHONDONTWRITEBYTECODE=1 TMPDIR="$W/tmp"
python -m ben.promotion.attempt3_train "$(dirname "$old_manifest")" "$new" \
  "$(dirname "$s1_checkpoint")" "$code/group-split.json" "$W/outputs" | tee "$W/progress.txt"
for m in s1 s2; do
  [[ -s "$W/outputs/attempt3-$m-held-out.npz" ]] || stop "missing $m predictions"
  for g in A B C D E final; do
    [[ -s "$W/outputs/ben-landcover-india-$m-$g.pt" ]] || stop "missing $m/$g weights"
  done
done
(cd "$W/outputs" && sha256sum G1.json *.npz *.pt > SHA256SUMS)
echo "CELL DONE"
