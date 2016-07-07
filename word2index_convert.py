from word2index import UNIGRAM_DICT_FILE
from preprocess.data import ReVerbPairs, word2index
import pickle as pkl
import os
import sys
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
DUMP_FILE = "./data/reverb-train.part%d.indx"

if __name__ == '__main__':
    # check if the index version exists
    for part in range(30):
        path = DUMP_FILE % part
        if os.path.exists(path):
            print("Index version data exists")
            sys.exit(0)

    # load vocabulary dictionary
    with open(UNIGRAM_DICT_FILE, 'rb') as f:
        voc_dict = pkl.load(f)

    # convert text to index
    for part in range(30):
        path = DUMP_FILE % part
        logging.info("converting part %d" % part)
        data = ReVerbPairs(usage='train', part=part, mode='str')

        line_num = 0
        with open(path, 'a') as f:
            length = len(data)
            for q, a in data:
                sys.stdout.write("\rLoad: %.2f" % (float(line_num/length)))
                sys.stdout.flush()

                q_indx = [str(word2index(token, voc_dict)) for token in q]
                a_indx = [str(word2index(token, voc_dict)) for token in a]
                new_q = " ".join(q_indx)
                new_a = " ".join(q_indx)
                f.write("%s\t%s\n" % (new_q, new_a))

                line_num += 1

