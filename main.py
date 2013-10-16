import oRSM
import cPickle
import re
import numpy as np

__author__ = "Dongwoo Kim"
__date__ = "2013/10/15"

def get_data(filepath, stopwordspath):
    """
    read artices from filepath (line by line)
    construct vocabulary without stopwords (from stopwordspath)
    very slow (need to modify)
    """
    minimum_doc_length = 10
    minimum_word_length = 2
    maximum_word_length = 10
    maximum_voca = 1200

    from collections import Counter
    cnt = Counter()
    data = []
    stopwords = set([word.strip() for word in " ".join(open(stopwordspath).readlines()).split()])
    with open(filepath) as rawinput:
        lines = rawinput.readlines()
        tmp = []
        voca = set()
        for line in lines:
            words = re.findall(r'\w{%d,%d}' % (minimum_word_length, maximum_word_length), line.lower())
            cnt.update(words)
            tmp.append(words)

        for key in stopwords:
            del cnt[key]

        voca = [key for key,val in cnt.most_common(maximum_voca)]
        voca = voca[201:]

        for doc in tmp:
            tmpdoc = [doc.count(word) for word in voca]
            if sum(tmpdoc) > minimum_doc_length:
                data.append(tmpdoc)
            print len(data)
    return data, voca


if __name__ == '__main__':
    #data, voca = get_data('content_sample.txt', 'stop-words-english1.txt')
    #cPickle.dump(data,open('sample_data.pcl','w'))
    #cPickle.dump(voca,open('sample_voca.pcl','w'))

    data = cPickle.load(open('sample_data.pcl'))
    voca = cPickle.load(open('sample_voca.pcl'))

    h = 10
    rsm = oRSM.RSM(h,len(voca))
    errors = rsm.train(1000, data, 0.1)
    #errors = rsm.train_minibatch(2000,data,0.1,128)
    #rsm = oRSM.oRSM(h,len(voca), 50)
    #errors = rsm.train_minibatch(1000, data, 0.1, 128)
    #np.savetxt('test.csv', rsm.W, header=','.join(voca), delimiter=',')

    cPickle

    #write top words when each hidden unit activated
    with open('top_words.csv','w') as topout:
        sorted_idx = []
        for i in range(h):
            hidden = np.zeros(h)
            hidden[i] = 1
            prob = rsm.visible_activation_probability(hidden)
            sorted_idx.append(prob.argsort())
            topout.write('topic' + str(i) +',')
        topout.write('\n')

        for i in range(10):
            for j in range(h):
                topout.write(voca[sorted_idx[j][-i-1]] + ",")
            topout.write('\n')

    with open('W.csv','w') as output:
        for i in range(rsm.K):
            output.write(voca[i])
            for j in range(h):
                output.write(',' + str(rsm.W[j,i]))
            output.write('\n')

    ### here for lda
    wordids = list()
    wordcts = list()
    for item in data:
        wordids.append([i for i in range(len(item)) if item[i] > 0])
        wordcts.append([item[i] for i in range(len(item)) if item[i] > 0])

    # import lda
    # import printtopics

    # alpha = 0.1
    # eta = 0.01
    # K = 10

    # _lda = lda.LDA(voca, K, wordids, wordcts, alpha, eta)

    # oldbound = 0
    # finalIter = 0
    # for iteration in xrange(100):
    #     (gamma, bound) = _lda.do_m_step( )
    #     if(abs(oldbound - bound) < 0.0001):
    #         finalIter = iteration
    #         break;
    #     oldbound = bound
    #     finalIter = iteration
    #     print iteration, bound

    # np.savetxt('lda_lambda-%d.dat' % finalIter, _lda._lambda)
    # np.savetxt('lda_gamma-%d.dat' % finalIter, gamma)

    # printtopics.print_topics(voca, 'lda_lambda-%d.dat' % (finalIter), 'lda_topics.txt')


