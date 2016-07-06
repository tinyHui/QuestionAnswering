import logging


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
UNIGRAM_DICT_FILE = './bin/unigram_indx_hash.pkl'
BIGRAM_DICT_FILE = './bin/bigram_indx_hash.pkl'
THRIGRAM_DICT_FILE = './bin/thrigram_indx_hash.pkl'
LOWEST_FREQ = 4


if __name__ == "__main__":
    from collections import defaultdict
    from preprocess.data import ReVerbPairs, UNKNOWN_TOKEN, UNKNOWN_TOKEN_INDX
    import pickle as pkl
    import os
    import sys
    import argparse

    parser = argparse.ArgumentParser(description='Define training process.')
    parser.add_argument('--gram', type=int, default=1, help='Define N for Ngram')
    args = parser.parse_args()

    gram = args.grams
    if gram == 1:
        DUMP_FILE = UNIGRAM_DICT_FILE
    elif gram == 2:
        DUMP_FILE = BIGRAM_DICT_FILE
    elif gram == 3:
        DUMP_FILE = THRIGRAM_DICT_FILE
    else:
        raise SystemError("Does not support %d-gram" % gram)

    if os.path.exists(DUMP_FILE):
        logging.info("Word index file exists, skip")
        sys.exit(0)

    multi_gram = gram > 1
    if multi_gram:
        assert os.path.exists(UNIGRAM_DICT_FILE), "Must train unigram first"

    logging.info("Generating source data")
    # data is a group of sentences
    src_data = ReVerbPairs(usage='train', mode='str')

    logging.info("Extracting tokens")
    logging.warning("Ignore tokens appears less than %d" % LOWEST_FREQ)
    token_count_group = {}
    token_group = defaultdict(list)
    line_num = 0
    for i in src_data.sent_indx:
        token_count_group[i] = defaultdict(int)
        for line in src_data:
            sys.stdout.write("\rLoad: %d/%d" % (line_num, len(src_data)))
            sys.stdout.flush()
            tokens = line[i]
            for token in zip(*[tokens[i:] for i in range(gram)]):
                # check if the token appears count reach requirement
                if token_count_group[i][token] > LOWEST_FREQ:
                    token_group[i].append(token)
                else:
                    token_count_group[i][token] += 1
            line_num += 1

    logging.info("Generating token dictionary")
    word_indx_hash_group = {}
    for i in src_data.sent_indx:
        # unique
        token_list = list(set(token_group[i]))
        # add unknown token
        token_group[i].insert(UNKNOWN_TOKEN_INDX, UNKNOWN_TOKEN)
        # generate index
        word_indx_hash_group[i] = dict(zip(token_list, range(len(token_list))))
        logging.info("Found %d tokens" % len(token_list))

    logging.info("Saving word index hashing table")
    with open(DUMP_FILE, 'wb') as f:
        pkl.dump(word_indx_hash_group, f)
