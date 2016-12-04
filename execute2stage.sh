#!/usr/bin/env bash
set -u
set -e

if [[ $1 =~ ^Y$|^$ ]]
then
  # get paragraph Q projector vector
    env/bin/python text2feature.py --feature avg --CCA_stage -1 --worker 40 --freq 2000
    env/bin/python train.py --feature avg --stage paraphrase --segment
fi

# train 2-stage CCA
env/bin/python text2feature.py --feature avg --CCA_stage 2 --worker 40 --freq 2000
env/bin/python train.py --feature avg --stage 2stage --segment

# test
env/bin/python test.py --CCA_stage 2 --feature avg ./bin/CCA.avg.pkl --para_map_file ./bin/ParaMap.avg.pkl
sh run_eval.sh ./data/questions.txt ./data/labels.txt ./result/reverb-test-with_dist.avg.txt > ./result/result.txt