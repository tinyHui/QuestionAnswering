if __name__ == '__main__':
    from preprocess.data import ReVerbPairs, ParaphraseWikiAnswer, ParaphraseMicrosoftRaw
    from word2vec import WORD_EMBEDDING_BIN_FILE
    import pickle as pkl

    with open(WORD_EMBEDDING_BIN_FILE, 'rb') as f:
        emb_voc_dict = pkl.load(f)

    reverb = ReVerbPairs(usage='train', mode='raw_token', grams=1)
    para_wiki = ParaphraseWikiAnswer(mode='raw_token', grams=1)
    para_msr = ParaphraseMicrosoftRaw(mode='raw_token', grams=1)

    check_pending_list = [
        ("Question Answer", reverb),
        ("WikiAnswer Paraphrase", para_wiki),
        ("MSR Paraphrase", para_msr),
    ]

    for description, src_data in check_pending_list:
        voc_num = 0
        unseen_num = 0

        voc_list = []
        for line in src_data:
            for i in src_data.sent_indx:
                for token in line[i]:
                    try:
                        _ = emb_voc_dict[token]
                    except KeyError:
                        unseen_num += 1
                    finally:
                        voc_list.append(token)
                        voc_num += 1

        print("Consider Appear Times: {}: {}/{}={}% unseen".format(description, unseen_num, voc_num, float(unseen_num) / voc_num * 100))

        voc_list = set(voc_list)
        unseen_num = 0
        voc_num = len(voc_list)
        for token in voc_list:
            try:
                _ = emb_voc_dict[token]
            except KeyError:
                unseen_num += 1
        print("Consider Only Token: {}: {}/{}={}% unseen".format(description, unseen_num, voc_num, float(unseen_num) / voc_num * 100))
