from collections import defaultdict
from sys import stdout
from preprocess.data import no_symbol

FILE = '/disk/ocean/s1516713/wikianswers-paraphrases-1.0/paraphrases.wikianswer.full.txt'

if __name__ == '__main__':
    # generate hash map
    # paraphrase questions mapping
    para_lemma_map = defaultdict(list)
    i = 0
    stdout.write("Loading wikianswers-paraphrases-1.0/word_alignments.txt\n")
    with open('/disk/ocean/s1516713/wikianswers-paraphrases-1.0/word_alignments.txt', 'r') as f:
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
    with open('/disk/ocean/s1516713/wikianswers-paraphrases-1.0/questions.txt', 'r') as f:
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
        with open('/disk/ocean/s1516713/wikianswers-paraphrases-1.0/questions.txt', 'r') as fr:
            for line in fr:
                content = line.strip()
                try:
                    # question, tokens, POS, lemma
                    _, q, _, lemma = content.split('\t')
                    # remove symbols
                    q = no_symbol(q)
                    # replace "' '" to ""
                    q = q.replace(" ' ' ", " ")
                    # uppercase first letter
                    q = q.capitalize()

                    # get corresponde lemma sentences
                    q_para_lemma_list = para_lemma_map[lemma]

                    # generate paraphrase pairs
                    for q_para_lemma in q_para_lemma_list:
                        # for each lemma sentence, find its original sentence
                        q_para = sent_lemma_map[q_para_lemma]

                        # remove symbols
                        q_para = no_symbol(q_para)
                        # remove symbols
                        q_para = q_para.replace(" ' ' ", " ")
                        # uppercase first letter
                        q_para = q_para.capitalize()

                        # record down
                        fw.write("{}\t{}\n".format(q, q_para))
                        i += 1
                        stdout.write("\rgenerated: %d" % i)
                        stdout.flush()
                except ValueError:
                    continue
                except KeyError:
                    continue
    stdout.write("\nTotal: %d\n" % i)
