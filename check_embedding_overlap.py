import sys
import pickle as pkl
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


if __name__ == '__main__':
    embedding_models_files = sys.argv[1:]
    assert len(embedding_models_files) > 1, "must have more than (include) 2 embedding models"

    embedding_models = []
    for model_file in embedding_models_files:
        logging.info("loading: %s" % model_file)
        with open(model_file, 'rb') as f:
            model = pkl.load(f)
        model_tokens = model.keys()
        embedding_models.append(model_tokens)
        logging.info("%s contains %d tokens" % (model_file, len(model_tokens)))

    # merge all tokens
    all_tokens = []
    for model_tokens in embedding_models:
        all_tokens += model_tokens
    all_tokens = set(all_tokens)
    logging.info("Total: %d tokens" % len(all_tokens))

    share_tokens_count = 0
    not_share_tokens_count = [0] * len(embedding_models)
    for token in all_tokens:
        is_share = True
        for i, model_tokens in enumerate(embedding_models):
            if token not in model_tokens:
                not_share_tokens_count[i] += 1
                is_share = False

        if is_share:
            share_tokens_count += 1

    print("Non-share Token Counts:")
    for f, count in zip(embedding_models_files, not_share_tokens_count):
        print("\t%s: %d" % (f, count))

    print("Share Token Counts: %d" % share_tokens_count)
