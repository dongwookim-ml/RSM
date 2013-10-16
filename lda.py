import numpy as np
import re, string
from scipy.special import gammaln, psi

meanchangethresh = 0.001

def dirichlet_expectation(alpha):
    if(len(alpha.shape) == 1):
        return(psi(alpha)-psi(np.sum(alpha)))
    return (psi(alpha)-psi(np.sum(alpha, 1))[:,np.newaxis])

def parse_doc_list(docs, vocab):
    """
    Parse a document into a list of word ids and a list of counts,
    or parse a set of documents into two lists of lists of word ids
    and counts.

    Arguments: 
    docs:  List of D documents. Each document must be represented as
            a single string. (Word order is unimportant.) Any
            words not in the vocabulary will be ignored.
    vocab: Dictionary mapping from words to integer ids.

    Returns a pair of lists of lists. 

    The first, wordids, says what vocabulary tokens are present in
    each document. wordids[i][j] gives the jth unique token present in
    document i. (Don't count on these tokens being in any particular
    order.)

    The second, wordcts, says how many times each vocabulary token is
    present. wordcts[i][j] is the number of times that the token given
    by wordids[i][j] appears in document i.
    """
    if (type(docs).__name__ == 'str'):
        temp = list()
        temp.append(docs)
        docs = temp

    D = len(docs)
    
    wordids = list()
    wordcts = list()
    for d in range(D):
        docs[d] = docs[d].lower()
        docs[d] = re.sub(r'-', ' ', docs[d])
        docs[d] = re.sub(r'[^a-z ]', '', docs[d])
        docs[d] = re.sub(r' +', ' ', docs[d])
        words = string.split(docs[d])
        ddict = dict()
        for word in words:
            if (word in vocab):
                wordtoken = vocab[word]
                if (not wordtoken in ddict):
                    ddict[wordtoken] = 0
                ddict[wordtoken] += 1
        wordids.append(ddict.keys())
        wordcts.append(ddict.values())

    return((wordids, wordcts))

class LDA:
    """
    Latent Dirichlet allocation with mean field variational inference
    """

    def __init__(self, vocab, K, wordids, wordcts, alpha, eta):
        self._vocab = dict()
        for word in vocab:
            #word = word.lower()
            #word = re.sub(r'[^a-z]', '', word)
            self._vocab[word] = len(self._vocab)
        self._W = len(vocab)
        self._K = K
        self._D = len(wordids)
        self._alpha = alpha
        self._eta = eta

        self._wordids = wordids
        self._wordcts = wordcts
        self._lambda = 1*np.random.gamma(100., 1./100, (self._K, self._W))
        self._Elogbeta = dirichlet_expectation(self._lambda)
        self._expElogbeta = np.exp(self._Elogbeta)
        
    def do_e_step(self):
        """
        compute approximate topic distribution of each document and each word
        """

        #random initialize gamma
        gamma = 1*np.random.gamma(100.,1./100, (self._D, self._K))
        Elogtheta = dirichlet_expectation(gamma)
        expElogtheta = np.exp(Elogtheta)

        #sufficient statistics to update lambda
        sstats = np.zeros(self._lambda.shape)

        for d in range(0, self._D):
            ids = self._wordids[d]
            cts = np.array(self._wordcts[d])
            gammad = gamma[d,:]

            Elogthetad = Elogtheta[d,:]
            expElogthetad = expElogtheta[d,:]
            expElogbetad = self._expElogbeta[:,ids]
            phinorm = np.dot(expElogthetad, expElogbetad) + 1e-100

            for it in xrange(100):
                lastgamma = gammad

                gammad = self._alpha + expElogthetad * \
                    np.dot(cts / phinorm, expElogbetad.T)

                Elogthetad = dirichlet_expectation(gammad)
                expElogthetad = np.exp(Elogthetad)
                phinorm = np.dot(expElogthetad, expElogbetad) + 1e-100

                meanchange = np.mean(abs(gammad - lastgamma))

                if(meanchange < meanchangethresh):
                    break

            gamma[d,:] = gammad
            sstats[:, ids] += np.outer(expElogthetad.T, cts/phinorm)

        # This step finishes computing the sufficient statistics for the
        # M step, so that
        # sstats[k, w] = \sum_d n_{dw} * phi_{dwk} 
        # = \sum_d n_{dw} * exp{Elogtheta_{dk} + Elogbeta_{kw}} / phinorm_{dw}.
        sstats = sstats * self._expElogbeta

        return (gamma,sstats)


    def do_m_step(self):
        """
        estimate topic distribution based on computed approx. topic distribution
        """
        (gamma, sstats) = self.do_e_step()

        self._lambda = self._eta + sstats
        self._Elogbeta = dirichlet_expectation(self._lambda)
        self._expElogbeta = np.exp(self._Elogbeta)

        bound = self.approx_bound(gamma)

        return (gamma,bound)

    def approx_bound(self, gamma):
        """
        Compute lower bound of the corpus
        """

        score = 0
        Elogtheta = dirichlet_expectation(gamma)
        expElogtheta = np.exp(Elogtheta)

        # E[log p(docs | theta, beta)]
        for d in range(0, self._D):
            gammad = gamma[d, :]
            ids = self._wordids[d]
            cts = np.array(self._wordcts[d])
            phinorm = np.zeros(len(ids))
            for i in range(0, len(ids)):
                temp = Elogtheta[d, :] + self._Elogbeta[:, ids[i]]
                tmax = max(temp)
                phinorm[i] = np.log(sum(np.exp(temp - tmax))) + tmax
            score += np.sum(cts * phinorm)

        # E[log p(theta | alpha) - log q(theta | gamma)]
        score += np.sum((self._alpha - gamma)*Elogtheta)
        score += np.sum(gammaln(gamma) - gammaln(self._alpha))
        score += sum(gammaln(self._alpha*self._K) - gammaln(np.sum(gamma, 1)))

        # E[log p(beta | eta) - log q (beta | lambda)]
        score = score + np.sum((self._eta-self._lambda)*self._Elogbeta)
        score = score + np.sum(gammaln(self._lambda) - gammaln(self._eta))
        score = score + np.sum(gammaln(self._eta*self._W) - gammaln(np.sum(self._lambda, 1)))

        return(score)