import sys
import pickle as pkl
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


if __name__ == '__main__':
    from preprocess.data import ReVerbPairs, ParaphraseWikiAnswer, ParaphraseMicrosoftRaw

    embedding_models_files = sys.argv[1:]
    assert len(sys.argv) > 1, "must provide at one embedding binary file"

    embedding_models = []
    for model_file in embedding_models_files:
        logging.info("loading: %s" % model_file)
        with open(model_file, 'rb') as f:
            model = pkl.load(f)
        embedding_models.append(model)

    reverb_train = ReVerbPairs(usage='test', mode='raw_token', grams=1)
    reverb_test = ReVerbPairs(usage='train', mode='raw_token', grams=1)
    para_wiki = ParaphraseWikiAnswer(mode='raw_token')
    # para_msr = ParaphraseMicrosoftRaw(mode='raw_token')

    check_pending_list = [
        ("Question Answer Train", reverb_train),
        ("Question Answer Test", reverb_test),
        ("WikiAnswer Paraphrase", para_wiki),
        # ("MSR Paraphrase", para_msr),
    ]

    for fname, emb_voc_dict in zip(embedding_models_files, embedding_models):
        print("For %s" % fname)
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

            print("\tConsider Appear Times: {}: {}/{}={}% unseen".format(description, unseen_num, voc_num, float(unseen_num) / voc_num * 100))

            voc_list = set(voc_list)
            unseen_num = 0
            voc_num = len(voc_list)
            for token in voc_list:
                try:
                    _ = emb_voc_dict[token]
                except KeyError:
                    unseen_num += 1
            print("\tConsider Only Token: {}: {}/{}={}% unseen".format(description, unseen_num, voc_num, float(unseen_num) / voc_num * 100))
