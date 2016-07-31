from preprocess.data import ReVerbTestRaw
from preprocess.feats import get_lemmas
from sys import stdout

FILE = './data/reverb-test.txt'

if __name__ == '__main__':
    qa = ReVerbTestRaw()
    prev_q = None
    prev_q_lemmas = None
    i = 0
    with open(FILE, 'a') as f:
        for q_id, q, a, l in qa:
            i += 1
            stdout.write("\rgenerated: %d" % i)
            stdout.flush()
            if q == prev_q:
                q_lemmas = prev_q_lemmas
            else:
                q_lemmas = get_lemmas(q_lemmas)
            f.write("{}\t{}\t{}\t{}\n".format(q_id, q_lemmas, a, l))
            prev_q = q
            prev_q_lemmas = q_lemmas
    stdout.write("\nTotal: %d\n" % i)