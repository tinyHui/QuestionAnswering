#!/usr/bin/env bash

set -u
set -e
set -v

################################
#
# train "main train"
#
################################

ln -s ../data_main_train ./data

# use Google News embedding
# 1 stage
rm -f ./bin
ln -s ../main_train_GoogleNews_1stage ./bin
sh execute1stage.sh

# 2 stage
rm -f ./bin
ln -s ../main_train_GoogleNews_2stage ./bin
sh execute2stage.sh


# use Giga embedding
# 1 stage
rm -f ./bin
ln -s ../main_train_Giga_1stage ./bin
sh execute1stage.sh

# 2 stage
rm -f ./bin
ln -s ../main_train_Giga_2stage ./bin
sh execute2stage.sh


# use GigaPara embedding
# 1 stage
ln -s ../main_train_GigaPara_1stage ./bin
sh execute1stage.sh

# 2 stage
rm -f ./bin
ln -s ../main_train_GigaPara_2stage ./bin
sh execute2stage.sh

rm -f ./data
rm -f ./bin

# show results
tail -n +1 ../main_train_*_*stage/result/*
