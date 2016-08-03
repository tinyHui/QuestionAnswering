from collections import defaultdict
import pprint
import sys

def load_labels(input):
    result = defaultdict(dict)
    for line in input:
        fields = line.rstrip().split('\t')
        if fields[0] == 'u': continue
        label = int(fields[0])
        question = fields[1]
        answer = tuple(fields[2].split(' '))
        result[question][answer] = label
    return result

def load_predictions(input):
    result = defaultdict(dict)
    for line in input:
        fields = line.rstrip().split('\t')
        question = fields[0]
        conf = float(fields[1])
        answer = tuple(fields[2].split(' '))
        result[question][answer] = conf
    return result

def load_questions(input):
    return list(line.strip() for line in input)

def avep(question, preds, labels):
    answers = preds[question]
    if not answers: return 0.0
    denom = sum(labels[question][a] for a in labels[question])
    if denom == 0.0: return 0.0
    corr = 0.0
    total = 0.0
    numer = 0.0
    for a in sorted(answers, key=answers.get, reverse=True):
        total += 1.0
        if a not in labels[question]:
            print >>sys.stderr, 'WARNING: no label for %s' % str(a)
            continue
        corr += labels[question][a]
        prec = corr / total
        numer += prec * labels[question][a]
    return numer / denom

def mean_avep(questions, preds, labels):
    n = float(len(questions))
    return sum(avep(q, preds, labels) for q in questions) / n

def at1_stats(questions, preds, labels):
    correct = 0.0
    num_pred = 0.0
    num_ranking_errors = 0.0
    num_lexicon_errors = 0.0
    total = 0.0
    for question in questions:
        total += 1
        answers = preds[question]
        if not answers:     
            num_lexicon_errors += 1
            continue

        # best_answer = max(answers, key=answers.get)
        # using distance instead of similarity
        best_answer = min(answers, key=answers.get)
        num_pred += 1
        if best_answer not in labels[question]:
            sys.stderr.write("WARNING: no label for %r" % str(best_answer))
            continue
        correct += labels[question][best_answer]
        if not labels[question][best_answer]:
            if any(a in labels[question] and labels[question][a] for a in answers):
                num_ranking_errors += 1
            else:
                num_lexicon_errors += 1
    p = correct/num_pred
    r = correct/len(questions)
    stats = {
        'precision@1': correct/num_pred,
        'recall@1': correct/len(questions),
        'ranking_error_rate': num_ranking_errors/total,
        'lexicon_error_rate': num_lexicon_errors/total,
        'correct': correct,
        'total': total,
        'predictions': num_pred,
        'f1': 2*p*r/(p+r)
    }
    return stats

def print_stats(questions, preds, labels):
    stats = at1_stats(questions, preds, labels)
    stats['mean average precision'] = mean_avep(questions, preds, labels)
    pprint.pprint(stats)

def print_output(questions, preds, labels):
    for q in questions:
        print(q)
        answers = sorted([(v,k) for (k,v) in preds[q].items()], reverse=True)
        print(len(answers), 'answers')
        predicted = set()
        for score, answer in answers:
            label = ' ' if labels[q][answer] else '*'
            print('%s\t%0.2f\t(%s %s %s)' % (label, score, answer[0], answer[1], answer[2]))
            predicted.add(answer)
        unpredicted = set(a for a in labels[q] if a not in predicted and labels[q][a])
        for a in sorted(unpredicted):
            print('?\t\t(%s %s %s)' % (a[0], a[1], a[2]))
        print

def print_output_top(questions, preds, labels):
    for q in questions:
#        print q
        answers = sorted([(v,k) for (k,v) in preds[q].items()], reverse=True)
#        print len(answers), 'answers'
        predicted = set()
        n=0
        for score, answer in answers:
            label = ' ' if labels[q][answer] else '*'
            #print '%s\t%0.2f\t(%s %s %s)' % (label, score, answer[0], answer[1], answer[2])
            print('%s\t%s(%s, %s)' % (q, answer[0], answer[1], answer[2]))
            n += 1
            predicted.add(answer)
            break
        if n == 0:
            print('%s\t'% q)
        unpredicted = set(a for a in labels[q] if a not in predicted and labels[q][a])
#        for a in sorted(unpredicted):
#            print '?\t\t(%s %s %s)' % (a[0], a[1], a[2])
#        print


if __name__ == '__main__':
    mode = sys.argv.pop(1)
    questions = load_questions(open(sys.argv.pop(1)))
    labels = load_labels(open(sys.argv.pop(1)))
    preds = load_predictions(open(sys.argv.pop(1)))
    modes = {
        'printstats': print_stats,
        'printoutput': print_output,
        'printoutputtop': print_output_top,
    }
    modes[mode](questions, preds, labels)
