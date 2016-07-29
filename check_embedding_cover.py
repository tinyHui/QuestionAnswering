if __name__ == '__main__':
    from preprocess.data import ReVerbPairs, ParaphraseParalexRaw
    from word2vec import WORD_EMBEDDING_BIN_FILE, LOW_FREQ_TOKEN_FILE
    import pickle as pkl

    with open(WORD_EMBEDDING_BIN_FILE, 'rb') as f:
        emb_voc_dict = pkl.load(f)

    with open(LOW_FREQ_TOKEN_FILE, 'rb') as f:
        low_freq_token_list = pkl.load(f)

    reverb = ReVerbPairs(usage='train', mode='str', grams=1)
    paraphrase = ParaphraseParalexRaw(mode='str', grams=1)

    check_pending_list = [
        ("Question Answer", reverb),
        ("Paraphrase", paraphrase),
    ]

    for description, src_data in check_pending_list:
        voc_num = 0
        unseen_num = 0

        for line in src_data:
            for i in src_data.sent_indx:
                for token in line[i]:
                    if token in low_freq_token_list:
                        continue
                    try:
                        _ = emb_voc_dict[token]
                    except KeyError:
                        unseen_num += 1
                    finally:
                        voc_num += 1

        print("{}: {}/{}={}% unseen".format(description, unseen_num, voc_num, float(unseen_num) / voc_num * 100))
