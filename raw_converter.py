DUMP_TRAIN_FILE = "./data/reverb-train.%s"
DUMP_TEST_FILE = "./data/reverb-test.%s"
DUMP_PARA_PARALEX_FILE = "./data/paraphrases.wikianswer.%s"
DUMP_PARA_MS_FILE = "./data/paraphrases.ms.%s"


if __name__ == '__main__':
    from preprocess.data import ReVerbPairs, ParaphraseWikiAnswer,\
        UNKNOWN_TOKEN_INDX, UNKNOWN_TOKEN
    from preprocess.feats import get_parse_tree
    import os
    import sys
    import argparse

    def word2index(w, hash_map):
        try:
            return hash_map[w]
        except KeyError:
            # unseen token
            return UNKNOWN_TOKEN_INDX

    def word2hash(w, hash_map):
        try:
            value = hash_map[w]
        except KeyError:
            # for unseen words, the embedding is the average \in R^Embedding_size
            value = hash_map[UNKNOWN_TOKEN]
        return '|'.join(map(str, value))


    mode_support = ['unigram', 'bigram', 'trigram', 'embedding', 'structure', 'raw']

    parser = argparse.ArgumentParser(description='Define mode to choose version for converting.')
    parser.add_argument('--mode', type=str,
                        help='Convert text version to index, embedding or structure?')
    args = parser.parse_args()
    mode = args.mode

    print("converting raw string file into %s" % mode)
    gram = 1
    # load dictionary
    if mode == mode_support[0]:
        print("loading vocabulary index")
        data_mode = 'index'
        gram = 1
    elif mode == mode_support[1]:
        print("loading vocabulary index")
        data_mode = 'index'
        suf = 'bi'
        gram = 2
    elif mode == mode_support[2]:
        print("loading vocabulary index")
        data_mode = 'index'
        suf = 'tri'
        gram = 3
    elif mode == mode_support[3]:
        print("loading embedding hash")
        data_mode = 'embedding'
        suf = 'emb'
    elif mode == mode_support[4]:
        data_mode = 'structure'
        suf = 'struct'
    elif mode == mode_support[5]:
        data_mode = 'raw_token'
        suf = 'raw'
    else:
        raise SystemError("mode must be %s" % ', '.join(mode_support))

    data_list = []
    # add train data
    data = ReVerbPairs(usage='train', mode=data_mode, grams=gram)
    path = DUMP_TRAIN_FILE % suf
    data_list.append((data, path))

    # add test data
    data = ReVerbPairs(usage='test', mode=data_mode, grams=gram)
    path = DUMP_TEST_FILE % suf
    data_list.append((data, path))

    if mode not in mode_support[0:3]:
        # not to index version
        # add paraphrase questions data
        data = ParaphraseWikiAnswer(mode=data_mode)
        path = DUMP_PARA_PARALEX_FILE % suf
        data_list.append((data, path))

    job_id = 0
    for data, path in data_list:
        line_num = 0
        print("converting %s" % path)
        if os.path.exists(path):
            # check if the index version exists
            print("index version data %s exists" % path)
            continue

        with open(path, 'w') as f:
            length = len(data)

            for d in data:
                sys.stdout.write("\rLoad: %.2f%%" % (float(line_num / length) * 100))
                sys.stdout.flush()
                line_num += 1
                param_num = len(d)

                for i in range(param_num):
                    if i in data.sent_indx:
                        if mode in mode_support[0:3]:
                            tokens = d[i]
                            sentence = " ".join(tokens)
                        elif mode == mode_support[3]:
                            tokens = ['|'.join(map(str, token)) for token in d[i]]
                            sentence = " ".join(tokens)
                        elif mode == mode_support[4]:
                            # mode == mode_support[3]:
                            if data.is_q_indx(i):
                                sentence = get_parse_tree(d[i], job_id)
                                job_id += 1
                            else:
                                sentence = d[i]
                        else:
                            tokens = d[i]
                            sentence = " ".join(tokens)
                    else:
                        sentence = d[i]
                    if i+1 != param_num:
                        f.write("%s\t" % sentence)
                    else:
                        f.write("%s" % sentence)
                f.write("\n")

        sys.stdout.write("\n")
