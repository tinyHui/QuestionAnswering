#!/bin/bash
set -u
set -e

echo "Enter 'total sample lines number', 'sample parts', 'file name', 'file line number'."
read total_sample_lines_num parts_num

filename='./data/reverb-train.full.emb'
file_lines_num=43133211

part_lines_num=$(echo "$total_sample_lines_num/$parts_num" | bc)
max_parts=$(echo "$file_lines_num/$part_lines_num" | bc)

count=1
for i in $(shuf -i 1-$max_parts -n $parts_num | sort -n); do
    start=$(echo "($i-1)*$part_lines_num+1" | bc);
    end=$(echo "($i*$part_lines_num)" | bc);
    echo "generating part$count ./data/reverb-train.part$count.emb: $start-$end"

    sed -n "${start},${end}p" $filename > "./data/reverb-train.part$count.emb";

    count=$(echo "$count+1" | bc);
done;