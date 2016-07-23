UNIGRAM_DICT_FILE = './bin/unigram_indx_hash.pkl'
BIGRAM_DICT_FILE = './bin/bigram_indx_hash.pkl'
THRIGRAM_DICT_FILE = './bin/thrigram_indx_hash.pkl'
LOWEST_FREQ = 3


if __name__ == "__main__":
    from collections import defaultdict
    from preprocess.data import ReVerbPairs, ParaphraseQuestionRaw, UNKNOWN_TOKEN, UNKNOWN_TOKEN_INDX
    import pickle as pkl
    import os
    import sys
    import argparse
    import logging

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    parser = argparse.ArgumentParser(description='Define training process.')
    parser.add_argument('--grams', type=int, default=1, help='Define N for Ngram')
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
    train_data = ReVerbPairs(usage='train', mode='str', grams=gram)
    para_data = ParaphraseQuestionRaw(mode='str', grams=gram)

    logging.info("Extracting tokens")
    logging.warning("Ignore tokens appears less than %d" % LOWEST_FREQ)
    token_count_group = {}
    token_group = defaultdict(list)

    for i in train_data.sent_indx:
        token_count_group[i] = defaultdict(int)

    length = len(train_data) + len(para_data)
    # extract tokens in train data
    line_num = 1
    for line in train_data:
        sys.stdout.write("\rLoad: %d/%d" % (line_num, length))
        sys.stdout.flush()
        for i in train_data.sent_indx:
            tokens = line[i]
            for token in tokens:
                token_count_group[i][token] += 1
                # check if the token appears count reaches requirement
                if token_count_group[i][token] > LOWEST_FREQ:
                    token_group[i].append(token)
        line_num += 1

    # extract tokens in paraphrase data, add into question
    for q1_tokens, q2_tokens, _ in para_data:
        sys.stdout.write("\rLoad: %d/%d" % (line_num, length))
        sys.stdout.flush()
        i = train_data.question_index
        for token in q1_tokens + q2_tokens:
            token_count_group[i][token] += 1
            if token_count_group[i][token] > LOWEST_FREQ:
                token_group[i].append(token)

    sys.stdout.write("\n")
    logging.info("Generating token dictionary")
    word_indx_hash_group = {}
    for i in train_data.sent_indx:
        # unique
        token_list = list(set(token_group[i]))
        # add unknown token
        token_list.insert(UNKNOWN_TOKEN_INDX, UNKNOWN_TOKEN)
        # generate index
        word_indx_hash_group[i] = dict(zip(token_list, range(len(token_list))))
        logging.info("Found %d tokens" % len(token_list))

    logging.info("Saving word index hashing table")
    with open(DUMP_FILE, 'wb') as f:
        pkl.dump(word_indx_hash_group, f, protocol=4)
