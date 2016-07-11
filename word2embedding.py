import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
WORD_EMBEDDING_FILE = './data/embedding.txt'
WORD_EMBEDDING_BIN_FILE = './bin/word_embedding.pkl'
EMBEDDING_SIZE = 300


# def generate_dictionary(voc_indx_map, w_emb_map):
#     embedding_dict = defaultdict(partial(np.zeros, EMBEDDING_SIZE))
#     for w, indx in voc_indx_map.items():
#         try:
#             emb = np.asarray(w_emb_map[w], dtype='float32')
#         except KeyError:
#             continue
#         embedding_dict[indx] = emb
#     return embedding_dict


if __name__ == "__main__":
    from collections import defaultdict, UserList
    from preprocess.data import WordEmbeddingRaw
    from word2index import UNIGRAM_DICT_FILE
    from functools import partial
    from gensim.models import Word2Vec
    from preprocess.data import ReVerbPairs
    import pickle as pkl
    import os
    import sys
    import numpy as np
    import multiprocessing

    if os.path.exists(WORD_EMBEDDING_BIN_FILE):
        logging.info("Word embedding dictionary file exists, skip")
        sys.exit(0)

    if not os.path.exists(WORD_EMBEDDING_FILE):
        logging.info("Embedding text file does not exist, generate one")
        sentences = UserList()
        line_num = 1
        # load ReVerb QA pairs, use index directly
        src_data = ReVerbPairs(usage='train', mode='index')
        for line in src_data:
            sys.stdout.write("\rLoad: %d/%d" % (line_num, len(src_data)))
            sys.stdout.flush()
            for i in src_data.sent_indx:
                sentences.append(' '.join(line[i]))
        line_num += 1
        sys.stdout.write("\n")
        # calculate embedding vector
        logging.info("Generating embedding vectors")
        model = Word2Vec(sentences, size=400, window=5, min_count=5, workers=multiprocessing.cpu_count())
        model.save_word2vec_format(WORD_EMBEDDING_FILE, binary=False)

    # logging.info("loading vocabulary index")
    # with open(UNIGRAM_DICT_FILE, 'rb') as f:
    #     voc_dict = pkl.load(f)

    logging.info("Generating source data")
    # data is a group of sentences
    src_data = WordEmbeddingRaw()

    #####
    # old code, need to load word index, then map to word
    #####
    # w_emb_map = {}
    # for w, emb in src_data:
    #     w_emb_map[w] = emb
    #
    # word_emb_hash_group = {}
    # for indx, voc_indx_map in voc_dict.items():
    #     word_emb_hash_group[indx] = generate_dictionary(voc_indx_map, w_emb_map)

    #####
    # new code, use word index to train directly
    #####
    word_emb_hash_group = {}
    for w, emb in src_data:
        word_emb_hash_group[w] = emb
    logging.info("Saving word embedding dictionary")
    with open(WORD_EMBEDDING_FILE, 'wb') as f:
        pkl.dump(word_emb_hash_group, f)
