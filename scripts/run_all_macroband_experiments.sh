#!/usr/bin/env bash

set -e

python3 scripts/run_macroband_sampling.py --v=ac --s=mh
python3 scripts/run_macroband_sampling.py --v=ac --s=mclmc
python3 scripts/run_macroband_sampling.py --v=ac --s=pathfinder

python3 scripts/run_macroband_sampling.py --v=dc --s=mh
python3 scripts/run_macroband_sampling.py --v=dc --s=mclmc
python3 scripts/run_macroband_sampling.py --v=dc --s=pathfinder
