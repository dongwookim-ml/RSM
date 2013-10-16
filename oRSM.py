import numpy as np

__author__ = "Dongwoo Kim"
__date__ = "2013/10/15"


def sigmoid(x):
    return 1/(1+np.exp(-x))

def sample_multinomial(prob, M):
    return np.random.multinomial(M, prob)

class RSM:
    def __init__(self, F, K):
        """
        F = number of first layer hidden units
        K = vocabulary size
        W = F x K
        """
        self.F = F
        self.K = K
        self.W = np.random.random(size=(self.F, self.K))
        self.a = np.random.random(self.F)
        self.b = np.random.random(self.K)

    def hidden_activation_probability(self, v):
        return sigmoid(np.dot(self.W, v) + sum(v) * self.a)

    def visible_activation_probability(self, h):
        prob = np.exp(np.dot(self.W.T, h) + self.b)
        return prob/prob.sum()

    def sample_hidden(self,v):
        prob = self.hidden_activation_probability(v)
        return np.random.uniform(size=self.F) < prob

    def sample_visible(self, h, sample_size):
        prob = self.visible_activation_probability(h)
        return np.random.multinomial(sample_size, prob)

    def train(self, max_iter, data, e_0):
        error_list = []
        for iteration in xrange(max_iter):
            learning_rate = e_0 / (1.0+(float(iteration)/float(max_iter)))
            reconstruction_error = 0
            for item in data:
                #positive phase, sample h
                h = self.sample_hidden(item)
                positive = np.outer(h, item)

                #negative phase, sample v and h again
                item_resample = self.sample_visible(h, sum(item))
                h2 = self.sample_hidden(item_resample)
                negative = np.outer(h2, item_resample)

                assert(positive.size == self.W.size)

                #compute CD_1 gradient
                self.W += learning_rate*(positive-negative)
                self.a += learning_rate*(h-h2)
                self.b += learning_rate*(item-item_resample)

                #reconstruction error between original document and sampled document
                reconstruction_error += np.square(item - item_resample).sum()
            print iteration, learning_rate, reconstruction_error
            error_list.append(reconstruction_error)
        return error_list

    def train_minibatch(self, max_iter, data, e_0, m_size):
        error_list = []
        for iteration in xrange(max_iter):
            learning_rate = e_0 / (1.0+(float(iteration)/float(max_iter)))
            reconstruction_error = 0
            positive = np.zeros([self.F, self.K])
            negative = np.zeros([self.F, self.K])

            for item_no in np.random.permutation(len(data))[:m_size]:
                item = data[item_no]
                #positive phase, sample h
                h = self.sample_hidden(item)
                positive += np.outer(h, item)

                #negative phase, sample v and h again
                item_resample = self.sample_visible(h, sum(item))
                h2 = self.sample_hidden(item_resample)
                negative += np.outer(h2, item_resample)

                assert(positive.size == self.W.size)

                #compute CD_1 gradient
                self.W = self.W + learning_rate*(positive-negative)

                #reconstruction error between original document and sampled document
                reconstruction_error += np.square(item - item_resample).sum()

            self.W = self.W + learning_rate*(positive-negative)
                
            print iteration, learning_rate, reconstruction_error
            error_list.append(reconstruction_error)
        return error_list    

class oRSM:
    def __init__(self, F, K, M):
        """
        F = number of first layer hidden units
        K = vocabulary size
        M = number of second layer hidden units
        W = F x K
        """

        self.F = F
        self.K = K
        self.M = M
        self.W = np.random.random(size=(self.F, self.K))

    def variationalActivationMu_1(self,v, mu2):
        # mu1 = F x 1, W = F x K, mu2 = K x 1
        mu1 = sigmoid(np.dot(self.W, v + self.M*mu2))
        assert (self.F == mu1.size)
        return mu1

    def variationalActivationMu_2(self,mu1):
        #mu2 = K x 1
        mu2 = np.exp(np.dot(self.W.T, self.M*mu1))
        mu2 = mu2/mu2.sum()
        return mu2

    def getVariationalParam(self,item):
        converge = False

        mu2 = np.random.random(self.K)

        while not converge:
            mu1 = self.variationalActivationMu_1(item, mu2)
            old_mu2 = mu2
            mu2 = self.variationalActivationMu_2(mu1)

            if (old_mu2 - mu2).sum() < epsilon:
                converge = True

        return mu1, mu2

    def sample_hidden1(self,mu1):
        return np.random.uniform(size = mu1.size) < mu1

    def resample_item(self, item, h1):
        prob = np.exp(np.dot(self.W.T, h1))
        prob = prob/prob.sum()
        return sample_multinomial(prob, sum(item))

    def train(self, max_iter, data, e_0):
        error_list = []
        for iteration in xrange(max_iter):
            learning_rate = e_0 / (1.0+(float(iteration)/float(max_iter)))
            reconstruction_error = 0
            for item in data:
                mu1, mu2 = self.getVariationalParam(item)
                h1 = self.sample_hidden1(mu1)
                h2 = sample_multinomial(mu2, self.M)

                positive = np.outer(h1, item + h2)

                item_resample = self.resample_item(item, h1)
                mu1_2, mu2_2 = self.getVariationalParam(item_resample)
                h1_2 = self.sample_hidden1(mu1_2)
                h2_2 = sample_multinomial(mu2_2, self.M)

                negative = np.outer(h1_2, item_resample + h2_2)

                assert(positive.size == self.W.size)

                self.W = self.W + learning_rate*(positive-negative)

                reconstruction_error += np.square(item - item_resample).sum()
            print iteration, learning_rate, reconstruction_error
            error_list.append(reconstruction_error)
        return error_list

    def train_minibatch(self, max_iter, data, e_0, m_size):
        error_list = []
        for iteration in xrange(max_iter):
            learning_rate = e_0 / (1.0+(float(iteration)/float(max_iter)))
            reconstruction_error = 0
            positive = np.zeros([self.F, self.K])
            negative = np.zeros([self.F, self.K])
            for item_no in np.random.permutation(len(data))[:m_size]:
                item = data[item_no]
                mu1, mu2 = self.getVariationalParam(item)
                h1 = self.sample_hidden1(mu1)
                h2 = sample_multinomial(mu2, self.M)

                positive += np.outer(h1, item + h2)

                item_resample = self.resample_item(item, h1)
                mu1_2, mu2_2 = self.getVariationalParam(item_resample)
                h1_2 = self.sample_hidden1(mu1_2)
                h2_2 = sample_multinomial(mu2_2, self.M)

                negative += np.outer(h1_2, item_resample + h2_2)

                assert(positive.size == self.W.size)
                reconstruction_error += np.square(item - item_resample).sum()

            self.W = self.W + learning_rate*(positive-negative)
                
            print iteration, learning_rate, reconstruction_error
            error_list.append(reconstruction_error)
        return error_list
