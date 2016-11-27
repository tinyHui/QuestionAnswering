import pickle as pkl
from preprocess.data import ReVerbPairs, ParaphraseWikiAnswer
from word2vec import WORD_EMBEDDING_BIN_FILE
from sys import stdout


if __name__ == '__main__':
    with open(WORD_EMBEDDING_BIN_FILE, 'rb') as f:
        embedding_dict = pkl.load(f)
        embedding_tokens = embedding_dict.keys()
        del embedding_dict

    data_list = [('./data/reverb-train.filter.txt', ReVerbPairs(usage='train', mode='token')),
                 ('./data/paraphrases.wikianswer.filter.txt', ParaphraseWikiAnswer(mode='token'))]

    i = 0
    for target_fname, data in data_list:
        name = str(data)
        length = len(data)

        with open(target_fname, 'w') as f:
            for line in data:
                i += 1
                stdout.write("\rProcessed %s: %.2f%%" % (name, float(i) / length * 100))
                stdout.flush()

                write_content = ""
                unseen_count = 0
                for sent_index in data.sent_indx:
                    for token in line[sent_index]:
                        if token not in embedding_tokens:
                            unseen_count += 1

                    write_content += ' '.join(line[sent_index]) + '\t'

                if unseen_count < 2:
                    f.write(write_content.strip() + '\n')
