#!/usr/bin/env bash
# Polarity-head round: after the faithful DEVA-style tuning finishes (its tuned config is needed), runs pol_round.py, then audit.
source ~/.venv/bin/activate
cd ~/work/msa-pilot/scripts
F=~/work/msa-pilot/outputs/full
ts() { echo "[$(date '+%F %T')] $*"; }
until grep -qE "DEVA_DENSE_DONE" $F/run_deva_dense.log 2>/dev/null; do sleep 60; done
ts "start"
python pol_round.py > $F/logs/pol_round.log 2>&1; ts "$(grep -E 'chosen|POL_ROUND_DONE' $F/logs/pol_round.log | tr '\n' ' ')"
python audit.py > $F/logs/audit_pol.log 2>&1; ts "POL_DONE"
