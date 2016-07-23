TOKEN_STRUCT_SPLITTER = "@"
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

    mode_support = ['index', 'embedding']

    parse_text_job_id = 0
    for mode in mode_support:
        print("converting raw string file into %s" % mode)

        if mode == mode_support[0]:
            suf = 'indx'
            with open(UNIGRAM_DICT_FILE % "qa", 'rb') as f:
                qa_voc_dict = pkl.load(f)
            with open(UNIGRAM_DICT_FILE % "para", 'rb') as f:
                para_voc_dict = pkl.load(f)
        else:
            suf = 'emb'
            with open(WORD_EMBEDDING_BIN_FILE, 'rb') as f:
                emb_voc_dict = pkl.load(f)
            qa_voc_dict = emb_voc_dict
            para_voc_dict = emb_voc_dict

        data_list = []
        # add train data
        data = ReVerbPairs(usage='train', mode='str')
        path = DUMP_TRAIN_FILE % suf
        data_list.append((path, data, qa_voc_dict))

        # add test data
        data = ReVerbPairs(usage='test', mode='str')
        path = DUMP_TEST_FILE % suf
        data_list.append((path, data, qa_voc_dict))

        # add paraphrase questions data
        data = ParaphraseQuestionRaw(mode='str')
        path = DUMP_PARA_FILE % suf
        data_list.append((path, data, para_voc_dict))

        # load vocabulary dictionary
        print("loading vocabulary index")

        for path, data, voc_dict in data_list:
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

            with open(path, 'a') as f:
                length = len(data)

                for d in data:
                    sys.stdout.write("\rLoad: %.2f%%" % (float(line_num / length) * 100))
                    sys.stdout.flush()
                    line_num += 1
                    param_num = len(d)
                    for i in range(param_num):
                        if i in data.sent_indx:
                            if is_reverb_test:
                                if i == 1:
                                    voc_hash = voc_dict[0]
                                else:
                                    # i == 2
                                    voc_hash = voc_dict[1]

                            else:
                                voc_hash = voc_dict[i]

                            tokens = [str(word_hash(token, voc_hash, mode)) for token in d[i]]
                            sentence = " ".join(tokens)
                            if mode == 'embedding':
                                parsetree = get_parse_tree(" ".join(d[i]), parse_text_job_id)
                                print(parsetree)
                                parse_text_job_id += 1
                                # concatenate token senteence and parsetree
                                # split by @
                                sentence = "%s%s%s" % (sentence, TOKEN_STRUCT_SPLITTER, parsetree)
                        else:
                            sentence = d[i]
                        if i+1 != param_num:
                            f.write("%s\t" % sentence)
                        else:
                            f.write("%s" % sentence)
                    f.write("\n")

            sys.stdout.write("\n")
