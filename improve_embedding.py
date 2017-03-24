import os
from preprocess.data import UNKNOWN_TOKEN
from preprocess.data import ReVerbPairs, ParaphraseWikiAnswer
from scipy.spatial.distance import cosine
from sklearn.cluster import KMeans
from collections import defaultdict
from sys import stdout
import pickle as pkl
import logging


def Combiner(token, emb, embedding_group):
    for other_token, other_emb in embedding_group:
        if token == other_token:
            continue
        else:
            yield (token, emb), (other_token, other_emb)


def calc_dist(paras):
    (_, emb), (other_token, other_emb) = paras
    return other_token, cosine(emb, other_emb)


def find_embedding(token, main_embedding, other_embedding, kmean, other_emb_group, top=5):
    emb = None
    expended = False
    try:
        emb = main_embedding[token]

    except KeyError:
        emb_in_other_embedding = other_embedding[token]

        label = kmean.predict(emb_in_other_embedding.reshape(1, -1))[0]

        tokens_with_distance = []
        sub_emb_group = other_emb_group[label]
        l = Combiner(token, emb_in_other_embedding, sub_emb_group)
        for paras in l:
            tokens_with_distance.append(calc_dist(paras))

        top_similar_tokens = [x for x, _ in sorted(tokens_with_distance, key=lambda x: x[1])[:top]]
        for top_token in top_similar_tokens:
            try:
                emb = main_embedding[top_token]
                expended = True
                break
            except KeyError:
                continue

    if emb is not None:
        return emb, expended
    else:
        raise KeyError


def improve_embedding(tokens, gn_embedding, other_embedding, kmean, other_emb_group):
    improved_embedding = {}
    found_count = 0
    expend_count = 0
    token_count = len(tokens)

    for indx, token in enumerate(tokens):
        stdout.write("\rTesting: %d/%d, found %d, expended %d" % (indx+1, token_count, found_count, expend_count))
        stdout.flush()

        try:
            emb, expended = find_embedding(token, gn_embedding, other_embedding, kmean, other_emb_group)
            improved_embedding[token] = emb

            found_count += 1
            if expended:
                expend_count += 1

        except KeyError:
            continue

    stdout.write("\n")
    return improved_embedding


def group_embeddings(other_embedding, dump_kmean_fname, dump_group_fname, GROUPS=30000):
    if os.path.exists(dump_kmean_fname):
        logging.info("Loading kmeans")
        with open(dump_kmean_fname, 'rb') as f:
            kmeans = pkl.load(f)
    else:
        logging.info("Calculating kmeans using %d groups" % GROUPS)
        X = list(other_embedding.values())
        kmeans = KMeans(n_clusters=30000, n_jobs=32).fit(X)
        with open(dump_kmean_fname, 'wb') as f:
            pkl.dump(kmeans, f, protocol=4)

    if os.path.exists(dump_group_fname):
        logging.info("Loading groups")
        with open(dump_group_fname, 'rb') as f:
            group = pkl.load(f)
    else:
        logging.info("Grouping using kmeans")
        group = defaultdict(list)
        for label, word_emb in zip(kmeans.labels_, other_embedding.items()):
            group[label].append(word_emb)
        with open(dump_group_fname, 'wb') as f:
            pkl.dump(group, f, protocol=4)
    return kmeans, group


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    GIGA_EMBEDDING = "/disk/ocean/s1516713/Giga_Embedding/unigram_embedding.pkl"
    GIGA_PARA_KMEAN = "/disk/ocean/s1516713/GigaPara_Embedding/kmean.pkl"
    GIGA_PARA_GROUP = "/disk/ocean/s1516713/GigaPara_Embedding/group.pkl"

    GIGA_PARA_EMBEDDING = "/disk/ocean/s1516713/GigaPara_Embedding/unigram_embedding.pkl"
    GIGA_KMEAN = "/disk/ocean/s1516713/Giga_Embedding/kmean.pkl"
    GIGA_GROUP = "/disk/ocean/s1516713/Giga_Embedding/group.pkl"

    GOOGLE_NEWS_EMBEDDING = "/disk/ocean/s1516713/GoogleNews_Embedding/unigram_embedding.pkl"

    IMPROVED_WITH_GIGA_EMBEDDING = "/disk/ocean/s1516713/Improved_Embedding/Giga/unigram_embedding.pkl"
    IMPROVED_WITH_GIGA_PARA_EMBEDDING = "/disk/ocean/s1516713/Improved_Embedding/GigaPara/unigram_embedding.pkl"
    TOKENS = "/disk/ocean/s1516713/data_actual_full/tokens.pkl"

    logging.info("Getting tokens")

    if os.path.exists(TOKENS):
        with open(TOKENS, 'rb') as f:
            tokens = pkl.load(f)
    else:
        tokens = []
        datas = [ReVerbPairs(usage='train', mode='raw_token'), ParaphraseWikiAnswer(mode='raw_token')]
        for data in datas:
            for s in data:
                for i in data.sent_indx:
                    tokens += s[i]

        tokens = set(tokens)

        with open(TOKENS, 'wb') as f:
            pkl.dump(tokens, f, protocol=4)

    token_count = len(tokens)
    logging.info("Found %d tokens" % token_count)

    # load embeddings
    logging.info("Loading Google Embedding")
    with open(GOOGLE_NEWS_EMBEDDING, 'rb') as f:
        gn_embedding = pkl.load(f)

    # get improved embedding using GigaPara embedding
    logging.info("Loading GigaPara Embedding")
    with open(GIGA_PARA_EMBEDDING, 'rb') as f:
        other_embedding = pkl.load(f)

    kmean, other_emb_group = group_embeddings(other_embedding, GIGA_PARA_KMEAN, GIGA_PARA_GROUP)
    improved_embedding = improve_embedding(tokens, gn_embedding, other_embedding, kmean, other_emb_group)

    improved_embedding[UNKNOWN_TOKEN] = gn_embedding[UNKNOWN_TOKEN]

    logging.info("Saving Improved embedding with Giga Para Embedding")
    with open(IMPROVED_WITH_GIGA_PARA_EMBEDDING, 'wb') as f:
        pkl.dump(improved_embedding, f, protocol=4)

    # get improved embedding using Giga embedding
    logging.info("Loading Giga Embedding")
    with open(GIGA_EMBEDDING, 'rb') as f:
        other_embedding = pkl.load(f)

    kmean, other_emb_group = group_embeddings(other_embedding, GIGA_KMEAN, GIGA_GROUP)
    improved_embedding = improve_embedding(tokens, gn_embedding, other_embedding, kmean, other_emb_group)

    improved_embedding[UNKNOWN_TOKEN] = gn_embedding[UNKNOWN_TOKEN]

    logging.info("Saving Improved embedding with Giga Embedding")
    with open(IMPROVED_WITH_GIGA_EMBEDDING, 'wb') as f:
        pkl.dump(improved_embedding, f, protocol=4)
