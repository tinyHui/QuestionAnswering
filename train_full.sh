#!/usr/bin/env bash

set -u
set -e
set -v

################################
#
# train "full"
#
################################

ln -s ../data_full ./data

# use Google News embedding
# 1 stage
ln -s ../full_GoogleNews_1stage ./bin
sh execute1stage.sh

# 2 stage
rm ./bin
ln -s ../full_GoogleNews_2stage ./bin
sh execute2stage.sh


# use Giga embedding
# 1 stage
ln -s ../full_Giga_1stage ./bin
sh execute1stage.sh

# 2 stage
rm ./bin
ln -s ../full_Giga_2stage ./bin
sh execute2stage.sh


# use GigaPara embedding
# 1 stage
ln -s ../full_GigaPara_1stage ./bin
sh execute1stage.sh

# 2 stage
rm ./bin
ln -s ../full_GigaPara_2stage ./bin
sh execute2stage.sh

rm -f ./data

################################
#
# train "full"
#
################################

ln -s ../data_actual_full ./data

# use Google News embedding
# 1 stage
ln -s ../actual_full_GoogleNews_1stage ./bin
sh execute1stage.sh

# 2 stage
rm ./bin
ln -s ../actual_full_GoogleNews_2stage ./bin
ln -s ../full_GoogleNews_2stage/ParaMap.avg.pkl ./bin/ParaMap.avg.pkl
sh execute2stage.sh N


# use Giga embedding
# 1 stage
ln -s ../actual_full_Giga_1stage ./bin
sh execute1stage.sh

# 2 stage
rm ./bin
ln -s ../actual_full_Giga_2stage ./bin
ln -s ../full_Giga_2stage/ParaMap.avg.pkl ./bin/ParaMap.avg.pkl
sh execute2stage.sh N


# use GigaPara embedding
# 1 stage
ln -s ../actual_full_GigaPara_1stage ./bin
sh execute1stage.sh

# 2 stage
rm ./bin
ln -s ../actual_full_GigaPara_2stage ./bin
ln -s ../full_GigaPara_2stage/ParaMap.avg.pkl ./bin/ParaMap.avg.pkl
sh execute2stage.sh N

rm ./data

# show results
tail -n +1 ../*full_*_*stage/result/*