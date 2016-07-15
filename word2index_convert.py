from word2index import UNIGRAM_DICT_FILE
from preprocess.data import ReVerbPairs, word2index
import pickle as pkl
import os
import sys
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
DUMP_TRAIN_FILE = "./data/reverb-train.part%d.indx"
DUMP_TEST_FILE = "./data/reverb-test.full.indx"

if __name__ == '__main__':
    from train import PROCESS_NUM
    # load vocabulary dictionary
    logging.info("loading vocabulary index")
    with open(UNIGRAM_DICT_FILE, 'rb') as f:
        voc_dict = pkl.load(f)

    data_list = []

    # add train data
    for part in range(PROCESS_NUM):
        path = DUMP_TRAIN_FILE % part
        if os.path.exists(path):
            # check if the index version exists
            print("Index version data %s exists" % path)
            continue
        logging.info("converting part %d" % part)
        data = ReVerbPairs(usage='train', part=part, mode='str')
        data_list.append(path, data)

    # add test data
    data = ReVerbPairs(usage='test', mode='str')
    data_list.append((DUMP_TEST_FILE, data))

    for path, data in data_list:
        line_num = 0
        with open(path, 'a') as f:
            length = len(data)
            for d in data:
                sys.stdout.write("\rLoad: %.2f%%" % (float(line_num / length) * 100))
                sys.stdout.flush()
                line_num += 1
                if data.usage == 'train':
                    q, a = d
                    q_indx = [str(word2index(token, voc_dict[0])) for token in q]
                    a_indx = [str(word2index(token, voc_dict[1])) for token in a]
                    new_q = " ".join(q_indx)
                    new_a = " ".join(a_indx)
                    f.write("%s\t%s\n" % (new_q, new_a))
                else:
                    q, a, q_id, l = d
                    q_indx = [str(word2index(token, voc_dict[0])) for token in q]
                    a_indx = [str(word2index(token, voc_dict[1])) for token in a]
                    new_q = " ".join(q_indx)
                    new_a = " ".join(a_indx)
                    f.write("%d\t%s\t%s\t%d\n" % (q_id, new_q, new_a, l))

            sys.stdout.write("\n")


