if __name__ == '__main__':
    from hash_index import UNIGRAM_DICT_FILE
    from word2vec import WORD_EMBEDDING_BIN_FILE
    import pickle as pkl

    with open(WORD_EMBEDDING_BIN_FILE, 'rb') as f:
        emb_voc_dict = pkl.load(f)
        emb_voc = emb_voc_dict.keys()

    with open(UNIGRAM_DICT_FILE % "qa", 'rb') as f:
        qa_voc_dict = pkl.load(f)
        qa_q_voc = qa_voc_dict[0].keys()
        qa_a_voc = qa_voc_dict[1].keys()

    with open(UNIGRAM_DICT_FILE % "para", 'rb') as f:
        para_voc_dict = pkl.load(f)
        para_q_voc = para_voc_dict[0].keys()
        para_a_voc = para_voc_dict[1].keys()

    check_pending_list = [
        ("Question Answer - Question vocabulary", qa_q_voc),
        ("Question Answer - Answer vocabulary", qa_q_voc),
        ("Paraphrase - Question vocabulary", para_q_voc),
        ("Paraphrase - Answer vocabulary", para_a_voc)
    ]

    for description, voc_list in check_pending_list:
        voc_num = len(voc_list)
        unseen_num = 0
        for voc in voc_list:
            try:
                _ = emb_voc[voc]
            except KeyError:
                unseen_num += 1

        print("{}: {}/{}={}% unseen".format(description, unseen_num, voc_num, float(unseen_num) / voc_num * 100))

