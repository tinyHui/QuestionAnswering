# How To Generate Data
## WikiAnswer Paraphrase
python generate_paraphrase.py
## MSR Paraphrase
grep “^1” msr_paraphrase_train.txt > msr_paraphrase.full.txt
grep “^1” msr_paraphrase_test.txt >> msr_paraphrase.full.txt

## Train & Test
Download paralex data set, copy these following files to ./data/:
1. <path_to_paralex>/data/*
2. <path_to_paralex>/data/tuples.db, rename to reverb-tuples.db

Execute the following commands:
```
python generate_reverb_train.py
python generate_reverb_test.py
```

Download pretrained GoogleNews-vectors-negative300.bin.gz from [word2vec project](https://code.google.com/archive/p/word2vec/). Then execute:
```
python -c '
from gensim.models import word2vec
model = word2vec.Word2Vec.load_word2vec_format('path/to/GoogleNews-vectors-negative300.bin', binary=True)
model.save_word2vec_format('./data/embeddings.txt', binary=False)
python hash_embedding.py
```

# How To Train & Test
## Train
We can use the following command to get the usage manual.
```
python train.py --help
python test.py --help
```
There are three options for the feature:
1. unigram
2. average word embedding (avg)
3. holographic circular correlation

N.B: before training CCA model, we need to make sure the return line number inside the ./preprocess/data.py is match the actual file number.
- ParaphraseWikiAnswer class \_\_len\_\_ function, use
```
wc -l ./data/paraphrases.wikianswer.txt
```
- ReVerbPairs class \_\_len\_\_ function ‘train’ if branch, use
```
wc -l ./data/reverb-train.txt
```

### 1-Stage CCA
```
python train.py --feature unigram --sparse --svds 300
```

### 2-Stage CCA
The following command will train the paraphrase mapping model.
```
python train.py --feature avg --CCA_stage -1
```
Then, we can apply the paraphrase mapping model in training. We can also reuse the feature representation matrix obtained in 1-Stage CCA.
```
python train.py --feature avg --CCA_stage 2 --para_map_file ./bin/ParaMap.pkl --reuse features ./bin/1_stage/Raw.avg.pkl
```

## Test
Let’s suppose we put trained model files under ./bin/1_stage and ./bin/2_stage
```
python test.py --CCA_stage 1 --feature avg ./bin/1_stage/CCA.avg.pkl
```

```
python test.py --CCA_stage 2 --feature avg ./bin/2_stage/CCA.avg.pkl --para_map_file ./bin/2_stage/ParaMap.pk
```

```
mv ./result/reverb-test-with_dist.avg.txt ./result/2_stage/reverb-test-with_dist.wikianswer.avg.txt
sh run_eval.sh ./data/questions.txt ./data/labels.txt ./result/2_stage/reverb-test-with_dist.wikianswer.avg.txt
```