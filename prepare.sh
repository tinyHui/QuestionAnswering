#!/usr/bin/env bash

# generate train, test data, paraphrase data
env/bin/python generate_reverb_train.py
env/bin/python generate_reverb_test.py
env/bin/python generate_paraphrase.py

# hash to index
env/bin/python hash_index.py --source qa
env/bin/python hash_index.py --source para

# hash to embedding
env/bin/python word2vec.py
env/bin/python hash_embedding.py