#!/usr/bin/env bash
# Runs all controlled experiments in sequence (about 12-14 h on one RTX 4000 Ada).
#   nohup ./run_all.sh > run_all.out 2>&1 &
# Quick check first (about 20 min, writes to results_smoke/):
#   LSB_SMOKE=1 LSB_RESULTS=results_smoke ./run_all.sh
cd "$(dirname "$0")"
export PYTHONIOENCODING=utf-8
export EPP_SEEDS="${EPP_SEEDS:-42,43,44,45,46}" EPP_SCENARIOS="${EPP_SCENARIOS:-CLEAN}"
export LSB_CIFAR_SEEDS="${LSB_CIFAR_SEEDS:-42,43,44,45,46}"
R="${LSB_RESULTS:-results}"; mkdir -p "$R/logs"; export LSB_RESULTS="$(cd "$R" && pwd)"
for job in transaction_simulation/run_dbh_matched.py \
           ecg_robustness/run_ecg_robust.py \
           cifar_epp/run_epp_matched.py \
           cifar_lsb/run_lsb_cifar.py; do
  name=$(basename "$(dirname "$job")"); echo "$(date)  start $name"
  python -u "$job" > "$LSB_RESULTS/logs/$name.log" 2>&1 || echo "  $name FAILED - see $LSB_RESULTS/logs/$name.log"
done
echo "$(date)  ALL DONE"
