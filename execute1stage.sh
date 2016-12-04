#!/usr/bin/env bash
set -u
set -e

# train 2-stage CCA
env/bin/python text2feature.py --feature avg --CCA_stage 1 --worker 40 --freq 2000
env/bin/python train.py --feature avg --stage 1stage --segment

# test
env/bin/python test.py --CCA_stage 1 --feature avg ./bin/CCA.avg.pkl
sh run_eval.sh ./data/questions.txt ./data/labels.txt ./result/reverb-test-with_dist.avg.txt > ./result/result.txt
