from collections import UserList, defaultdict
from preprocess.data import QAs
import logging
import pickle as pkl
import os
import sys

VOC_DICT_FILE = './bin/word_indx_hash.pkl'


if __name__ == "__main__":
    if os.path.exists(VOC_DICT_FILE):
        logging.info("Word index file exists, skip")
        sys.exit(0)

    # remove previous processed files
    logging.info("Clean up")
    if os.path.exists(VOC_DICT_FILE):
        os.remove(VOC_DICT_FILE)

    logging.info("Generating source data")
    # data is a group of sentences
    token_list = UserList()
    src_data = [QAs(mode='str')]
    for d in src_data:
        for s in d:
            for i in d.sent_indx:
                token_list += s[i]

    # generate dictionary
    logging.info("Generating token dictionary")
    unique_token_list = set(token_list)
    logging.info("Found %d tokens" % len(unique_token_list))
    word_indx_hash = defaultdict(int)
    for i, token in enumerate(unique_token_list):
        word_indx_hash[token] = i

    logging.info("Saving word index hashing table")
    with open(VOC_DICT_FILE, 'wb') as f:
        pkl.dump(word_indx_hash, f)

    logging.info("Free up memory")
    del token_list
    del unique_token_list
    del word_indx_hash
