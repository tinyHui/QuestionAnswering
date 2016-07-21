DUMP_TRAIN_FILE = "./data/reverb-train.%s"
DUMP_TEST_FILE = "./data/reverb-test.%s"
DUMP_PARA_FILE = "./data/paraphrases.%s"


if __name__ == '__main__':
    from hash_index import UNIGRAM_DICT_FILE
    from word2vec import WORD_EMBEDDING_BIN_FILE
    from preprocess.data import ReVerbPairs, ParaphraseQuestionRaw, UNKNOWN_TOKEN_INDX
    from word2vec import EMBEDDING_SIZE
    import pickle as pkl
    import os
    import sys
    import argparse
    import logging

    def word_hash(w, hash_map, mode):
        if mode == 'index':
            try:
                return hash_map[w]
            except KeyError:
                # unseen token
                return UNKNOWN_TOKEN_INDX
        elif mode == 'embedding':
            try:
                value = hash_map[w]
            except KeyError:
                # for unseen words, the embedding is zero \in R^Embedding_size
                value = [0] * EMBEDDING_SIZE

            return '|'.join(map(str, value))

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    parser = argparse.ArgumentParser(description='Define training process.')
    parser.add_argument('--mode', type=str)

    args = parser.parse_args()
    mode = args.mode
    mode_support = ['index', 'embedding']
    if mode not in mode_support:
        raise SystemError("mode only supports %s" % ",".join(mode_support))

    # load vocabulary dictionary
    logging.info("loading vocabulary index")
    if mode == mode_support[0]:
        with open(UNIGRAM_DICT_FILE, 'rb') as f:
            voc_dict = pkl.load(f)
        suf = 'indx'
    elif mode == mode_support[1]:
        with open(WORD_EMBEDDING_BIN_FILE, 'rb') as f:
            voc_dict = pkl.load(f)
        suf = 'emb'

    data_list = []
    # add train data
    data = ReVerbPairs(usage='train', mode='str')
    path = DUMP_TRAIN_FILE % suf
    data_list.append((path, data))

    # add test data
    data = ReVerbPairs(usage='test', mode='str')
    path = DUMP_TEST_FILE % suf
    data_list.append((path, data))

    # add paraphrase questions data
    data = ParaphraseQuestionRaw(mode='str')
    path = DUMP_PARA_FILE % suf
    data_list.append((path, data))

    for path, data in data_list:
        line_num = 0
        logging.info("converting %s" % path)
        if os.path.exists(path):
            # check if the index version exists
            logging.info("index version data %s exists" % path)
            continue
        with open(path, 'a') as f:
            length = len(data)

            if isinstance(data, ReVerbPairs):
                if data.usage == 'test':
                    voc_dict[2] = voc_dict[1]
                    voc_dict[1] = voc_dict[0]
            if isinstance(data, ParaphraseQuestionRaw):
                # paraphrase question data use question tokens
                voc_dict[1] = voc_dict[0]

            for d in data:
                sys.stdout.write("\rLoad: %.2f%%" % (float(line_num / length) * 100))
                sys.stdout.flush()
                line_num += 1
                param_num = len(d)
                for i in range(param_num):
                    if i in data.sent_indx:
                        tokens = [str(word_hash(token, voc_dict[i], mode)) for token in d[i]]
                        sentence = " ".join(tokens)
                    else:
                        sentence = d[i]
                    if i+1 != param_num:
                        f.write("%s\t" % sentence)
                    else:
                        f.write("%s" % sentence)
                f.write("\n")

        sys.stdout.write("\n")
