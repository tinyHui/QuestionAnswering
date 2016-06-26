from collections import UserList, UserDict
from preprocess.data import ReVerbPairs
import logging
import pickle as pkl
import os
import sys

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
VOC_DICT_FILE = './bin/word_indx_hash.pkl'


def generate_dictionary(token_list):
    # generate dictionary
    logging.info("Generating token dictionary")
    unique_token_list = set(token_list)
    logging.info("Found %d tokens" % len(unique_token_list))

    word_indx_hash = UserDict()
    for i, token in enumerate(unique_token_list):
        word_indx_hash[token] = i + 1

    del unique_token_list
    return word_indx_hash


if __name__ == "__main__":
    if os.path.exists(VOC_DICT_FILE):
        logging.info("Word index file exists, skip")
        sys.exit(0)

    logging.info("Generating source data")
    # data is a group of sentences
    src_data = ReVerbPairs(usage='train', mode='str')

    word_indx_hash_group = {}
    for i in src_data.sent_indx:
        token_list = UserList()
        for line in src_data:
            token_list += line[i]
        word_indx_hash_group[i] = generate_dictionary(token_list)
        del token_list[:]

    logging.info("Saving word index hashing table")
    with open(VOC_DICT_FILE, 'wb') as f:
        pkl.dump(word_indx_hash_group, f)

    logging.info("Free up memory")
    del word_indx_hash_group
