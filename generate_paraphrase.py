from collections import defaultdict
from sys import stdout
from preprocess.data import process_raw

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

    i = 0
    stdout.write("Generating paraphrases.wikianswer.txt\n")
    with open(FILE, 'a') as fw:
        with open('./data/wikianswers-paraphrases-1.0/questions.50000.txt', 'r') as fr:
            for line in fr:
                content = line.strip()
                try:
                    # question, tokens, POS, lemma
                    _, q, _, lemma = content.split('\t')
                    # find paraphrase sentences, lemma version
                    q_para_lemma_list = para_lemma_map[lemma]
                    for q_para_lemma in q_para_lemma_list:
                        # for each lemma sentence, find its original sentence
                        q_para = sent_lemma_map[q_para_lemma]
                        # generalize the sentence, remember to add ? in the end
                        q_proc = process_raw(q) + ' ?'
                        q_para_proc = process_raw(q_para) + ' ?'
                        # record down
                        fw.write("{}\t{}\n".format(q_proc, q_para_proc))
                        i += 1
                        stdout.write("\rgenerated: %d" % i)
                        stdout.flush()
                except ValueError:
                    continue
                except KeyError:
                    continue
    stdout.write("\nTotal: %d\n" % i)
