import pickle as pkl
from preprocess.data import ReVerbPairs, ParaphraseWikiAnswer
from word2vec import WORD_EMBEDDING_FILE
from sys import stdout


if __name__ == '__main__':
    with open(WORD_EMBEDDING_FILE, 'rb') as f:
        embedding_dict = pkl.load(f)
        embedding_tokens = embedding_dict.keys()
        del embedding_dict

    data_list = [('./data/reverb.train.filter.txt', ReVerbPairs(usage='train', mode='proc_token')),
                 ('./data/paraphrases.wikianswer.filter.txt', ParaphraseWikiAnswer(mode='proc_token'))]

    i = 0
    for target_fname, data in data_list:
        name = str(data)
        length = len(data)

        with open(target_fname, 'w') as f:
            for line in data:
                write_content = ""
                for sent_index in data.sent_indx:
                    i += 1
                    stdout.write("\rProcessed %s: %.2f%%" % (name, float(i) / length))
                    stdout.flush()

                    unseen_count = 0
                    for token in line[sent_index]:
                        if token not in embedding_tokens:
                            unseen_count += 1

                        if unseen_count > 1:
                            continue

                    write_content += ' '.join(line[sent_index]) + '\t'

                f.write(write_content.strip() + '\n')



