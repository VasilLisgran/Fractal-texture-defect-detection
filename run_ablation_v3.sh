#!/bin/bash
# Ablation of Table 11 of the paper (one reference, main design v3): the four variants that
# run_core_v3.sh and run_extra_v3.sh do not produce. Run after run_core_v3.sh, from the repository root:
#   bash run_ablation_v3.sh        # about 4 x 6 minutes on a laptop CPU
# The other rows of the table come from run_core_v3.sh: plain and full = ours_s0_1.json, light = light_s0_1.json.
# The parameters below are those stored in the "meta" field of the original score files v3res/abl_*.json.

DATA=~/aitex_v3
RES=v3res
CATS="fabric_00 fabric_01 fabric_02 fabric_03 fabric_04 fabric_06"
S0=$DATA/s0/1shot
mkdir -p $RES/logs
[ -d $S0 ] || { echo "No $S0: run run_core_v3.sh first"; exit 1; }

run() { local name=$1; shift; echo "[$(date +%H:%M:%S)] $name"; "$@" > $RES/logs/$name.log 2>&1 || echo "   !!! error, see $RES/logs/$name.log"; }
OURS="python3 eval_mvtec.py --cat $CATS --root $S0 --shots 1 --methods affine --mean-w 0"

# smax 0 means no clipping of the contrast at all (both bounds off, see eval_mvtec.py)
run abl_1 $OURS --smin 1   --smax 1   --save-scores $RES/abl_1_center_only.json           # contrast fixed to 1 (centring only)
run abl_2 $OURS --smin 0.6 --smax 0                                    --save-scores $RES/abl_2_fit_unbounded.json        # fit, unbounded, no normalisation
run abl_3 $OURS --smin 0.6 --smax 0   --norm --save-scores $RES/abl_3_fit_unbounded_norm.json   # fit, unbounded, normalised
run abl_4 $OURS --smin 0.6 --smax 1.7                                  --save-scores $RES/abl_4_fit_bounded_nonorm.json   # fit, bounded, no normalisation
echo "Done. Intervals: bash optional_intervals.sh > intervals.txt"
