#!/bin/bash
set -u
set -e
if [ $# != 3 ]; then
    echo "Usage: run_eval.sh questions labels predictions"
    echo "Where"
    echo "    questions = path to questions file (one per line)"
    echo "    labels = path to labels file e.g. data/questions/labels.txt"
    echo "    predictions = path to output e.g. output of run_qa.sh"
    echo 
    echo "Computes metrics of the given output evaluated against the given"
    echo "labels."
    echo
    exit 1
fi
questions=$1
labels=$2
predictions=$3
PYTHONPATH=$PWD/python python eval.py printstats $questions $labels $predictions
