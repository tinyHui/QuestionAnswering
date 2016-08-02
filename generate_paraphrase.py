from collections import defaultdict
from sys import stdout
from preprocess.data import process_raw
from word2vec import WORD_EMBEDDING_BIN_FILE
from random import sample
import pickle as pkl

FILE = './data/paraphrases.wikianswer.txt'

if __name__ == '__main__':
    # generate hash map
    # paraphrase questions mapping
    para_lemma_map = defaultdict(list)
    i = 0
    stdout.write("Loading wikianswers-paraphrases-1.0/word_alignments.txt\n")
    with open('./data/wikianswers-paraphrases-1.0/word_alignments.txt', 'r') as f:
        for line in f:
            content = line.strip()
            q1, q2, _ = content.split('\t')
            para_lemma_map[q1].append(q2)
            i += 1
            stdout.write("\rloaded: %.2f%%" % (float(i) / 35291309 * 100))
            stdout.flush()
    stdout.write("\n")

    # origin sentence - lemma mapping
    sent_lemma_map = {}
    i = 0
    stdout.write("Loading wikianswers-paraphrases-1.0/questions.txt\n")
    with open('./data/wikianswers-paraphrases-1.0/questions.txt', 'r') as f:
        for line in f:
            content = line.strip()
            try:
                # question, tokens, POS, lemma
                _, q, _, lemma = content.split('\t')
                sent_lemma_map[lemma] = q
                i += 1
                stdout.write("\rloaded: %.2f%%" % (float(i) / 2585750 * 100))
                stdout.flush()
            except ValueError:
                continue
    stdout.write("\n")

    stdout.write("Loading word embedding hashmap\n")
    with open(WORD_EMBEDDING_BIN_FILE, 'rb') as f:
        emb_voc_dict = pkl.load(f)
    word_embedding_keys = emb_voc_dict.keys()

    i = 0
    stdout.write("Generating paraphrases.wikianswer.txt\n")
    with open(FILE, 'a') as fw:
        with open('./data/wikianswers-paraphrases-1.0/questions.txt', 'r') as fr:
            for line in fr:
                if i > 300000:
                    break

                content = line.strip()
                try:
                    # question, tokens, POS, lemma
                    _, q, _, lemma = content.split('\t')
                    q_tokens = q.split()
                    # make sure all tokens have the embedding
                    if any([token not in word_embedding_keys for token in q_tokens]):
                        continue
                    # find paraphrase sentences, lemma version
                    try:
                        q_para_lemma_list = sample(para_lemma_map[lemma], 3)
                    except ValueError:
                        # sample number larger than population
                        q_para_lemma_list = para_lemma_map[lemma]
                    # generate paraphrase pairs
                    for q_para_lemma in q_para_lemma_list:
                        # for each lemma sentence, find its original sentence
                        q_para = sent_lemma_map[q_para_lemma]
                        # generalize the sentence, remember to add ? in the end
                        # record down
                        fw.write("{}\t{}\n".format(q.lower(), q_para.lower()))
                        i += 1
                        stdout.write("\rgenerated: %d" % i)
                        stdout.flush()
                except ValueError:
                    continue
                except KeyError:
                    continue
    stdout.write("\nTotal: %d\n" % i)
