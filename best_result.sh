#!/bin/bash
set -u
set -e

echo "Enter 'sample parts'"
read parts_num

for i in $(seq $parts_num); do
    echo "training part$i ./data/reverb-train.part$i.emb"

    mv "./data/reverb-train.part$i.emb" ./data/reverb-train.emb
    python train.py --feature avg --CCA_stage 2 --para_map_file /disk/ocean/s1516713/bin5/ParaMap.pkl

    mv ./data/reverb-train.emb "./data/reverb-train.part$i.emb"
    mv ./bin/Raw.avg.pkl "/disk/ocean/s1516713/bin5/Raw.avg.part$i.pkl"
    mv ./bin/XCOV.avg.pkl "/disk/ocean/s1516713/bin5/XCOV.avg.part$i.pkl"
    mv ./bin/CCA.avg.pkl "/disk/ocean/s1516713/bin5/CCA.avg.part$i.pkl"
done;

for i in $(seq $parts_num); do
    echo "testing part$i"

    python test.py "--feature avg /disk/ocean/s1516713/bin5/CCA.avg.part$i.pkl" --CCA_stage 2 --para_map_file /disk/ocean/s1516713/bin5/ParaMap.pkl
done;