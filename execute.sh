set -u
set -e
set -v

source env/bin/activate

# generate train, test data
# python generate_reverb_train.py
# python generate_reverb_test.py

# hash to index
# python hash_index.py --source qa
# python hash_index.py --source para

# hash to embedding
# python word2vec.py
python hash_embedding.py

# get paragraph Q projector vector
python text2feature.py --feature avg --CCA_stage -1 --worker 40 --freq 2000
python train.py --feature avg --stage 1stage

# train 2-stage CCA
python text2feature.py --feature avg --CCA_stage 2 --worker 40 --freq 2000
python train.py --feature avg --stage 2stage

# test
python test.py --CCA_stage 2 --feature avg ./bin/2_stage/CCA.avg.pkl --para_map_file ./bin/2_stage/ParaMap.pkl
sh run_eval.sh ./data/questions.txt ./data/labels.txt ./result/2_stage/reverb-test-with_dist.avg.txt