from preprocess.data import ReVerbTrainRaw
import codecs
from sys import stdout

FILE = './data/reverb-train.txt'

if __name__ == '__main__':
    data = ReVerbTrainRaw()
    i = 1
    with codecs.open(FILE, 'a', 'utf-8') as f:
        for q, a in data:
            stdout.write("\rgenerated: %d" % i)
            stdout.flush()
            f.write("{}\t{}\n".format(q, a))
            i += 1