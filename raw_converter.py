DUMP_TRAIN_FILE = "./data/reverb-train.full.%s"
DUMP_TEST_FILE = "./data/reverb-test.full.%s"


if __name__ == '__main__':
    from train import PROCESS_NUM
    from hash_index import UNIGRAM_DICT_FILE
    from preprocess.data import ReVerbPairs, UNKNOWN_TOKEN_INDX
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

            return '|'.join(value)

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    parser = argparse.ArgumentParser(description='Define training process.')
    parser.add_argument('--mode', type=str)

    args = parser.parse_args()
    mode = args.mode
    mode_support = ['index', 'embedding']
    if mode not in mode_support:
        raise SystemError("mode only supports %s" % ",".join(mode_support))

    if mode == mode_support[0]:
        suf = 'indx'
    elif mode == mode_support[1]:
        suf = 'emb'

    # load vocabulary dictionary
    logging.info("loading vocabulary index")
    with open(UNIGRAM_DICT_FILE, 'rb') as f:
        voc_dict = pkl.load(f)

    data_list = []
    # add train data
    data = ReVerbPairs(usage='train', mode='str')
    path = DUMP_TRAIN_FILE % suf
    data_list.append((path, data))

    # add test data
    data = ReVerbPairs(usage='test', mode='str')
    path = DUMP_TEST_FILE % suf
    data_list.append((path, data))

    for path, data in data_list:
        line_num = 0
        logging.info("converting part %d" % path)
        if os.path.exists(path):
            # check if the index version exists
            print("Index version data %s exists" % path)
        with open(path, 'a') as f:
            length = len(data)
            for d in data:
                sys.stdout.write("\rLoad: %.2f%%" % (float(line_num / length) * 100))
                sys.stdout.flush()
                line_num += 1
                if data.usage == 'train':
                    q, a = d
                    q_indx = [str(word_hash(token, voc_dict[0], mode)) for token in q]
                    a_indx = [str(word_hash(token, voc_dict[1], mode)) for token in a]
                    new_q = " ".join(q_indx)
                    new_a = " ".join(a_indx)
                    f.write("%s\t%s\n" % (new_q, new_a))
                else:
                    q, a, q_id, l = d
                    q_indx = [str(word_hash(token, voc_dict[0], mode)) for token in q]
                    a_indx = [str(word_hash(token, voc_dict[1], mode)) for token in a]
                    new_q = " ".join(q_indx)
                    new_a = " ".join(a_indx)
                    f.write("%d\t%s\t%s\t%d\n" % (q_id, new_q, new_a, l))

            sys.stdout.write("\n")


