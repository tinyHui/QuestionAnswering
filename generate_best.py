from word2vec import WORD_EMBEDDING_BIN_FILE
from preprocess.data import UNKNOWN_TOKEN
import pickle as pkl
import sys

DESIRED_TRAIN = 3000000

if __name__ == '__main__':

    with open(WORD_EMBEDDING_BIN_FILE, 'rb') as f:
        emb_voc_dict = pkl.load(f)

    # generate reverb-train
    print("Generating ./data/reverb-train.txt")
    reverb_train_full = open('./data/reverb-train.full.txt', 'r')
    reverb_train = open('./data/reverb-train.txt', 'w')

    count = 0
    for line in reverb_train_full:
        if count >= DESIRED_TRAIN:
            break

        content = line.strip()
        _, a_triple = content.split("\t")
        e1, r, e2 = a_triple.split('|')
        tokens = e1.split(' ') + r.split(' ') + e2.split(' ')

        if all([token in emb_voc_dict for token in tokens]):
            reverb_train.write(line)
            count += 1
            sys.stdout.write("\rGenerated: %d" % count)
            sys.stdout.flush()

    reverb_train_full.close()
    reverb_train.close()
    sys.stdout.write("\n")

    # generate paraphrase
    print("Generating ./data/paraphrases.wikianswer.txt")
    para_full = open('/disk/ocean/s1516713/wikianswers-paraphrases-1.0/paraphrases.wikianswer.full.txt', 'r')
    para = open('./data/paraphrases.wikianswer.txt', 'w')

    count = 0
    for line in para_full:
        content = line.strip()
        p1, p2 = content.split("\t")
        tokens = p1.split(' ') + p2.split(' ')

        if all([token in emb_voc_dict for token in tokens]):
            para.write(line)
            count += 1
            sys.stdout.write("\rGenerated: %d" % count)
            sys.stdout.flush()

    para_full.close()
    para.close()
    sys.stdout.write("\n")
