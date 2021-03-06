from preprocess.data import ReVerbTrainRaw
import codecs
from sys import stdout

FILE = './data/reverb-train.txt'

if __name__ == '__main__':
    data = ReVerbTrainRaw()
    i = 0
    with open(FILE, 'a') as f:
        for q, a in data:
            i += 1
            stdout.write("\rgenerated: %d" % i)
            stdout.flush()
            f.write("{}\t{}\n".format(q, a))
    stdout.write("\nTotal: %d\n" % i)