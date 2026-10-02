#!/usr/bin/env bash
# Regenerate and execute every example notebook in place, one at a time (under a memory cap where systemd is available).
# (Plotly static export uses Chromium, which hangs under `ulimit -v`; a cgroup cap is used instead.)
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
KERNEL=${KERNEL:-python3}
"$PY" scripts/make_notebooks.py
CAP=()
if command -v systemd-run >/dev/null 2>&1; then
  CAP=(systemd-run --user --scope -q -p MemoryMax=5G -p MemorySwapMax=0)
fi
for nb in notebooks/*.ipynb; do
  echo "== $nb"
  ( cd notebooks && "${CAP[@]}" \
      env MPLBACKEND=module://matplotlib_inline.backend_inline \
      "$PY" -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=600 \
      --ExecutePreprocessor.kernel_name="$KERNEL" "$(basename "$nb")" )
done
