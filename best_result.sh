#!/bin/bash
set -u
set -e

echo "Enter 'total sample lines number', 'sample parts', 'file name', 'file line number'."
read total_sample_lines_num parts_num

filename='./data/reverb-train.full.txt'
file_lines_num=43133211

part_lines_num=$(echo "$total_sample_lines_num/$parts_num" | bc)
max_parts=$(echo "$file_lines_num/$part_lines_num" | bc)

count=1
for i in $(shuf -i 1-$max_parts -n $parts_num | sort -n); do
    echo "generating part$count ./data/reverb-train.part$count.txt"
    start=$(echo "($i-1)*$part_lines_num+1" | bc);
    end=$(echo "($i*$part_lines_num)" | bc);
    sed -n "${start},${end}p" $filename > "./data/reverb-train.part$count.txt";
    count=$(echo "$count+1" | bc);
done;

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