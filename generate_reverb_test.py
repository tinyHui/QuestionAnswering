from preprocess.data import ReVerbTestRaw
from sys import stdout

FILE = './data/reverb-test.txt'

if __name__ == '__main__':
    qa = ReVerbTestRaw()
    i = 0
    with open(FILE, 'a') as f:
        for q_id, q, a, l in qa:
            i += 1
            stdout.write("\rgenerated: %d" % i)
            stdout.flush()
            f.write("{}\t{}\t{}\t{}\n".format(q_id, q, a, l))
    stdout.write("\nTotal: %d\n" % i)