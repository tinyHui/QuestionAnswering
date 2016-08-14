#!/bin/bash
set -u
set -e

for i in $(seq $parts_num); do
    echo "training part$i ./data/reverb-train.part$i.txt"

    ln -s "./data/reverb-train.part$i.txt" ./data/reverb-train.txt
    python train.py --feature avg --CCA_stage 2 --para_map_file /disk/ocean/s1516713/bin5/ParaMap.pkl

    rm "./data/reverb-train.part$i.txt"
    mv ./bin/Raw.avg.pkl "/disk/ocean/s1516713/bin5/Raw.avg.part$i.pkl"
    mv ./bin/XCOV.avg.pkl "/disk/ocean/s1516713/bin5/XCOV.avg.part$i.pkl"
    mv ./bin/CCA.avg.pkl "/disk/ocean/s1516713/bin5/CCA.avg.part$i.pkl"
done;

for i in $(seq $parts_num); do
    echo "testing part$i"

    python test.py "--feature avg /disk/ocean/s1516713/bin5/CCA.avg.part$i.pkl" --CCA_stage 2 --para_map_file /disk/ocean/s1516713/bin5/ParaMap.pkl
done;