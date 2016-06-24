from preprocess.data import ReVerbRaw
import codecs
from sys import stdout

FILE = './data/reverb-train'

if __name__ == '__main__':
    data = ReVerbRaw()
    i = 0
    with codecs.open(FILE, 'a', 'utf-8') as f:
        for q_tokens, a_tokens in data:
            stdout.write("\rgenerated: %d" % i)
            stdout.flush()
            q = ' '.join(q_tokens)
            a = ' '.join(a_tokens)
            f.write("{}\t{}\n".format(q, a))
            i += 1