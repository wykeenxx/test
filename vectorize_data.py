# -*- coding: utf-8 -*

from load_data import load_data
import collections
import jieba
import numpy as np
import matplotlib.pyplot as plt
import os

class Preprocess():
    """
    x: 1d list of strings with shape (m, )
    y: 2d list of strings (required unravel further) (m,?) (note: current expected 1 element in each instance)
    process x and y into bag of words and one hot label respectively
    """
    def __init__(self,texts,labels,most_common):

        assert len(texts) == len(labels)

        self.texts = texts      # list of string

        # label processing
        self.labels = self.unravel_label(labels)     # 2d list -> 1d list
        self.labels_onehot, self.label_to_idx = self.label_to_onehot(self.labels) # take into consideration all labels
        del labels

        # input text processing
        self.tokenized_texts = self.tokenize(texts)  # jieba parse
        del texts
        self.vocab_to_keep = self.most_common(self.tokenized_texts, most_common)  # eliminate uncommon word
        self.word_to_idx, self.idx_to_word = self.build_vocabdict(self.vocab_to_keep)
        # print self.word_to_idx
        self.text_bow = self.convert_bag_of_word(self.tokenized_texts, self.word_to_idx)

        print type(self.text_bow)
        print type(self.labels_onehot)
        print self.text_bow.shape
        print self.labels_onehot.shape

    def process_test(self,testdata):
        """
        transform test data from previous fit prior
        :param testdata:
        :return:
        """
        raise NotImplementedError

    def get_xy(self):
        return self.text_bow, self.labels_onehot

    def print_shape(self,array):
        """
        print shape of 2d-list by converting it to 2d-array
        :param array: 2d-list
        :return:
        """
        print 'shape = {} '.format(self.to_array(array).shape)


    def to_array(self,array):
        """
        turn 2d list into 2d-array
        :param array:
        :return:
        """
        return np.array([np.array(i) for i in array])

    def label_to_onehot(self,lbls,keep_all=True):
        """
        first convert word's label (e.g. [touteng,fashao,...]) into index e.g.[5,2,...]; then convert to 1 hot
        note: expect label with 1 class only ; expect unravel label
        :param lbl: 1d list of word , shape:(m,)
        :param keep_all: whether to convert rare class with occurences under threshold into 1 bucket or not
        :return: numpy array of onehot labels; np.shape(m,number of classes)
        """
        # raise exception if under expectation
        for lbl in lbls:
            if type(lbl) == list:
                raise NotImplementedError

        if keep_all:
            # build label_to_idx dictionary
            label_to_idx = {lbl:i for i,lbl in enumerate(set(lbls))}  # to save
            label_id = [label_to_idx[lbl] for lbl in lbls] # [5,2,...]

            # initialize one-hot np array with shape of (m, n_class)
            shape_label_oh = (len(label_id),len(label_to_idx))
            label_oh = np.zeros(shape=shape_label_oh,dtype=np.float32)

            #fill one-hot
            for i,lbl in enumerate(label_id):
                label_oh[i,lbl] = 1.

            return label_oh, label_to_idx

    def unravel_label(self,Labels):
        """
        convert from shape (m,1) to (m,)
        [[2],[3]] -> [2,3]
        :param Labels:
        :return:
        """
        return [label[0] for label in Labels]

    def convert_bag_of_word(self, parsed_text, word_to_idx):
        """
        convert tokenized text into bag of words
        :param parsed_text:
        :param word_to_idx:
        :return: shape:(m,vocab) (e.g. for 3 example of 8 vocabs [[5, 2, 1, 0, 1, 2, 0, 0], [20, 3, 2, 2, 3, 2, 0, 3], [0, 0, 0, 0, 0, 0, 0, 0]])
        """
        bow = [[0 for i in range(len(word_to_idx.keys()))] for j in range(len(parsed_text))]

        vocab = set(word_to_idx.keys())
        # print vocab
        for i,sent in enumerate(parsed_text):
            for word in sent:
                if word in vocab:
                    bow[i][word_to_idx[word]] += 1
        # print len(bow)
        # print bow
        # print len(bow[97])
        return self.to_array(bow)

    def build_vocabdict(self, vocabs):
        """
        build dictionary from a vocabulary set
        :param vocabs: required to be set
        :return:
        """
        word_to_idx = {vocab:i for i,vocab in enumerate(vocabs)}
        idx_to_word = dict((idx,word) for word,idx in word_to_idx.items())
        return word_to_idx,idx_to_word

    def flatten(self,dataset):
        """
        flatten input(list of list) into one list in order to feed in Counter for extracting features(top_k/most common)
        [[2,3],[4,4,5]] -> [2,3,4,4,5]
        :param dataset: list of list (text or label)
        :return: list
        """
        flattened = list()
        for instance in dataset:
            for i in instance:
                flattened.append(i)

        return flattened

    def most_common(self,parsed_text, n_vocab_keep):
        """
        flatten input parsed text -> load into counter -> keep most common
        :param parsed_text: list of sequences
        :param n_vocab_keep:
        :return: list of n most common word (e.g. [u'\uff0c', u'\u4e86', u'\u7684'] for n=3)
        """

        word_counter = collections.Counter(self.flatten(parsed_text))
        print('n of vocabulary in corpus = {}'.format(len(word_counter.keys())))
        return list(zip(*word_counter.most_common(n_vocab_keep))[0])


    def tokenize(self,input_text):
        """
        jieba parse (including punctuation)
        【'你好吗'，'我很好'】-》【【你好，吗】，【我，很，好】】
        :param input_text: list of strings
        :return: list of sequences
        """
        # time consuming...
        return [jieba.lcut(text) for text in input_text]

    def print_output_distribution(self):
        # to observe output/label distribution
        # plot counter
        output = collections.Counter(self.flatten(self.labels))

        temp = [-i for i in sorted([-i for i in output.values()])]
        print temp
        print 'n of classes: {}'.format(len(temp))
        del temp

def train_val_split(x,y,split_ratio,shuffle_seed=22):
    """
    n of examples (1st dimension) must be equal

    :param x: must be np.array
    :param y: must be np.array
    :return:
    """
    assert type(x) == np.ndarray
    assert type(y) == np.ndarray
    assert x.shape[0] == y.shape[0]  # n of examples must be same for x and y

    n_examples = y.shape[0]

    # generate shuffle index
    index = np.arange(n_examples)
    np.random.seed(shuffle_seed)  # initial 22
    np.random.shuffle(index)
    print('verify seed by logging index 20:{} '.format(index[20]))

    bound = int(split_ratio*index.shape[0])
    print bound
    assert len(set(index[:bound]) & set(index[bound:])) == 0 # no intersection of train/test allowed
    x_train = x[index[:bound]]
    y_train = y[index[:bound]]
    x_val = x[index[bound:]]
    y_val = y[index[bound:]]

    return x_train, y_train, x_val, y_val

def save(x_save,y_save, path):
    # not generalize, to be modified
    if not os.path.exists(path):
        os.mkdir(path)
        np.save('{}/x'.format(path), x_save)
        np.save('{}/y'.format(path), y_save)

    else:
        np.save('{}/x'.format(path), x_save)
        np.save('{}/y'.format(path), y_save)

    print ('save in {}'.format(path))

def load(path, x_fname = 'x.npy',y_fname = 'y.npy'):
    # bug: forgot to add '.npy' in load path name

    x_path = '{}/{}'.format(path, x_fname)
    y_path = '{}/{}'.format(path, y_fname)

    print ('load {}'.format(path))

    return np.load(x_path), np.load(y_path)


if __name__ == '__main__':

    import time
    start = time.time()
    DATAPATH = '/home/data/healthData/healthData/files/data/120ask/extracted/'
    DATAFILE = 'qa_trainset.json'
    texts,labels = load_data(datapath=DATAPATH, filename=DATAFILE, keep_only_single_output=True)


    preprocess = Preprocess(texts,labels,most_common=9000)
    x, y = preprocess.get_xy()

    save(x,y,path='np_processed_data') # save x and y

    # x,y = load(path='np_processed_data')

    # x_train, y_train, x_val, y_val = train_val_split(x, y, 0.8)
    # print x_train.shape
    # print y_train.shape
    # print x_val.shape
    # print y_val.shape


    end = time.time()
    print('execution time in {} minutes'.format((end-start)/60))
    # preprocess = Preprocess(texts,labels)
    # preprocess.print_output_distribution()


    # print preprocess.texts[10]
    # print preprocess.labels[10][0]


