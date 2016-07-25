DUMP_TRAIN_FILE = "./data/reverb-train.%s"
DUMP_TEST_FILE = "./data/reverb-test.%s"
DUMP_PARA_FILE = "./data/paraphrases.%s"


if __name__ == '__main__':
    from hash_index import UNIGRAM_DICT_FILE
    from word2vec import WORD_EMBEDDING_BIN_FILE
    from preprocess.data import ReVerbPairs, ParaphraseQuestionRaw, UNKNOWN_TOKEN_INDX
    from preprocess.feats import get_parse_tree
    from word2vec import EMBEDDING_SIZE
    import pickle as pkl
    import os
    import sys
    import argparse

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

    mode_support = ['index', 'embedding', 'structure']

    parser = argparse.ArgumentParser(description='Define source file to build index dictionary.')
    parser.add_argument('--mode', type=str,
                        help='Convert text version to index, embedding or structure?')
    args = parser.parse_args()
    mode = args.mode
    assert mode in mode_support, "mode must be %s" % ', '.join(mode_support)

    print("converting raw string file into %s" % mode)
    # load dictionary
    if mode == mode_support[0]:
        print("loading vocabulary index")
        data_mode = 'str'
        suf = 'indx'
        with open(UNIGRAM_DICT_FILE % "qa", 'rb') as f:
            qa_voc_dict = pkl.load(f)
        with open(UNIGRAM_DICT_FILE % "para", 'rb') as f:
            para_voc_dict = pkl.load(f)
    elif mode == mode_support[1]:
        print("loading embedding hash")
        data_mode = 'str'
        suf = 'emb'
        with open(WORD_EMBEDDING_BIN_FILE, 'rb') as f:
            emb_voc_dict = pkl.load(f)
    else:
        data_mode = 'raw'
        suf = 'struct'

    data_list = []
    # add train data
    data = ReVerbPairs(usage='train', mode=data_mode)
    path = DUMP_TRAIN_FILE % suf
    data_list.append((data, path))

    # add test data
    data = ReVerbPairs(usage='test', mode=data_mode)
    path = DUMP_TEST_FILE % suf
    data_list.append((data, path))

    # add paraphrase questions data
    data = ParaphraseQuestionRaw(mode=data_mode)
    path = DUMP_PARA_FILE % suf
    data_list.append((data, path))

    job_id = 0
    for data, path in data_list:
        line_num = 0
        print("converting %s" % path)
        if os.path.exists(path):
            # check if the index version exists
            print("index version data %s exists" % path)
            continue

        is_reverb_test = False
        if isinstance(data, ReVerbPairs):
            if data.get_usage() == 'test':
                is_reverb_test = True

        voc_hash = {}
        if mode == mode_support[0]:
            if isinstance(data, ReVerbPairs):
                if data.get_usage() == 'train':
                    voc_hash = qa_voc_dict
                elif data.get_usage() == 'test':
                    # for the reverb test data, each iteration return 4 items,
                    # q, a are located in index 1 and 2
                    voc_hash[1] = qa_voc_dict[0]
                    voc_hash[2] = qa_voc_dict[1]
            elif isinstance(data, ParaphraseQuestionRaw):
                voc_hash = para_voc_dict

        with open(path, 'a') as f:
            length = len(data)

            for d in data:
                sys.stdout.write("\rLoad: %.2f%%" % (float(line_num / length) * 100))
                sys.stdout.flush()
                line_num += 1
                param_num = len(d)

                for i in range(param_num):
                    if i in data.sent_indx:
                        if mode == mode_support[0]:
                            tokens = [str(word_hash(token, voc_hash[i], mode)) for token in d[i]]
                            sentence = " ".join(tokens)
                        elif mode == mode_support[1]:
                            tokens = [str(word_hash(token, emb_voc_dict, mode)) for token in d[i]]
                            sentence = " ".join(tokens)
                        else:
                            # mode == mode_support[2]:
                            if data.is_q_indx(i):
                                sentence = get_parse_tree(d[i], job_id)
                                job_id += 1
                            else:
                                sentence = d[i]
                    else:
                        sentence = d[i]
                    if i+1 != param_num:
                        f.write("%s\t" % sentence)
                    else:
                        f.write("%s" % sentence)
                f.write("\n")

        sys.stdout.write("\n")
