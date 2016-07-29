if __name__ == "__main__":
    from preprocess.data import ReVerbPairs, UNKNOWN_TOKEN
    from word2vec import WORD_EMBEDDING_BIN_FILE, EMBEDDING_SIZE, LOW_FREQ_TOKEN_FILE
    from preprocess.data import WordEmbeddingRaw
    from collections import defaultdict
    import pickle as pkl
    import numpy as np
    import os
    import sys
    import logging

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    if os.path.exists(WORD_EMBEDDING_BIN_FILE):
        logging.info("Word embedding dictionary file exists, skip")
        sys.exit(0)

    # the word embedding text file use raw word
    src_data = WordEmbeddingRaw()

    logging.info("converting raw embedding text")
    word_emb_hash = {}

    line_num = 1
    for w, emb in src_data:
        sys.stdout.write("\rLoad: %d/%d, %.2f%%" % (line_num, len(src_data), line_num/len(src_data)*100))
        sys.stdout.flush()
        line_num += 1
        word_emb_hash[w] = emb
    sys.stdout.write("\n")

    #
    #  calculate embedding for UNKNOWN token
    #
    # get token occur time
    logging.info("calculating embedding for UNKNOWN token")
    token_occur_count = defaultdict(int)
    src_data = ReVerbPairs(usage='train', mode='str', grams=1)
    for line in src_data:
        for i in src_data.sent_indx:
            for token in line[i]:
                token_occur_count[token] += 1

    unknown_emb = np.zeros(EMBEDDING_SIZE, dtype='float32')
    unknown_count = 0
    low_freq_token_list = []
    for token, occur_count in token_occur_count.items():
        if occur_count <= 3:
            try:
                # for token only occur once and token have trained embedding
                unknown_emb += word_emb_hash[token]
                low_freq_token_list.append(token)
                unknown_count += 1
            except KeyError:
                continue
    # get average of embedding
    unknown_emb /= float(unknown_count)
    word_emb_hash[UNKNOWN_TOKEN] = unknown_emb

    logging.info("Saving word embedding dictionary")
    with open(LOW_FREQ_TOKEN_FILE, 'wb') as f:
        pkl.dump(low_freq_token_list, f, protocol=4)

    with open(WORD_EMBEDDING_BIN_FILE, 'wb') as f:
        pkl.dump(word_emb_hash, f, protocol=4)
