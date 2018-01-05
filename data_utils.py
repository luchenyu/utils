import gensim, os, re, sys
import numpy as np

WORD2VEC = None

class Vocab(object):
    def __init__(self, filename, init=None, cutoff=None):
        self.filename = filename
        self.reserved_vocab_list = []
        self.vocab_list = []
        self.vocab_dict = {}
        if os.path.exists(filename) and init == None:
            for line in open(filename, 'r'):
                [key, count] = line.strip("\n").split("\t")
                if int(count) < 0:
                    self.reserved_vocab_list.append(key)
                self.vocab_list.append(key)
                self.vocab_dict[key] = [len(self.vocab_dict), int(count)]
        elif init != None:
            for item in init:
                self.reserved_vocab_list.append(item)
                self.vocab_list.append(item)
                self.vocab_dict[item] = [len(self.vocab_dict), -1]
        self.changed = False
        self.cutoff = cutoff if cutoff else sys.maxsize

    def idx2key(self, idx):
        """given index return key"""
        if idx >= min(self.cutoff, len(self.vocab_list)):
            return None
        else:
            return self.vocab_list[idx]

    def key2idx(self, key):
        """given key return index"""
        value = self.vocab_dict.get(key)
        if value:
            if value[0] < self.cutoff:
                return value[0]
            else:
                return None
        else:
            return None

    def size(self):
        """return size of the vocab"""
        return min(self.cutoff, len(self.vocab_list))

    def dump(self):
        """dump the vocab to the file"""
        if self.changed:
            with open(self.filename, 'w') as f:
                for key in self.vocab_list:
                    f.write(key+'\t'+str(self.vocab_dict[key][1])+'\n')

    def update(self, patch):
        """update the vocab"""
        self.changed = True
        for key in patch:
            if self.vocab_dict.has_key(key):
                if self.vocab_dict[key][1] >= 0:
                    self.vocab_dict[key][1] += patch[key]
            else:
                self.vocab_dict[key] = [len(self.vocab_dict), patch[key]]
        self.vocab_list = self.reserved_vocab_list + sorted(
            filter(lambda i: self.vocab_dict[i][1] >= 0, self.vocab_dict),
            key=lambda i: self.vocab_dict[i][1],
            reverse=True)
        for idx, item in enumerate(self.vocab_list):
            self.vocab_dict[item][0] = idx


    def sentence_to_token_ids(self, text):
        """encode a sentence in plain text into a sequence of token ids
        """
        if not type(text) is list:
            text = text.strip()
            text = map(lambda i: i.encode('utf-8'), list(text.decode('utf-8')))
        seq = [self.key2idx(key) for key in text]
        seq = [idx if idx else self.key2idx("_UNK") for idx in seq]
        return seq

    def token_ids_to_sentence(self, token_ids):
        """decode a sequence of token ids to a sentence
        """
        token_ids = filter(lambda i: i != self.key2idx("_PAD"), token_ids)
        token_ids = map(lambda i: i if self.idx2key(i) else self.key2idx("_UNK"), token_ids)
        text = "".join([self.idx2key(i) for i in token_ids])
        return text

class FastWord2vec(object):
    """
    Load word2vec model using gensim, cache the embedding matrix using numpy
    and memory-map it so that future loads are fast.
    """
    def __init__(self, path):
        if not os.path.exists(path + ".npy"):
            # load gensim
            word2vec = gensim.models.KeyedVectors.load_word2vec_format(
                path,
                binary=False)

            # save as numpy
            np.save(path + ".npy", word2vec.syn0)
            # also save the vocab
            with open(path + ".vocab", "wt") as fout:
                for word in word2vec.index2word:
                    fout.write(word.encode('utf-8') + "\n")

        self.syn0 = np.load(path + ".npy", mmap_mode="r")
        self.index2word = [l.strip("\n") for l in open(path + ".vocab", "rt")]
        self.word2index = {word: k for k, word in enumerate(self.index2word)}
        self._word_ending_tables = {}
        self._word_beginning_tables = {}

    def __getitem__(self, key):
        return np.array(self.syn0[self.word2index[key]])

    def __contains__(self, key):
        return key in self.word2index

    def words_ending_in(self, word_ending):
        if len(word_ending) == 0:
            return self.index2word
        self._build_word_ending_table(len(word_ending))
        return self._word_ending_tables[len(word_ending)].get(word_ending, [])

    def _build_word_ending_table(self, length):
        if length not in self._word_ending_tables:
            table = {}
            for word in self.index2word:
                if len(word) >= length:
                    ending = word[-length:]
                    if ending not in table:
                        table[ending] = [word]
                    else:
                        table[ending].append(word)
            self._word_ending_tables[length] = table

    def words_starting_in(self, word_beginning):
        if len(word_beginning) == 0:
            return self.index2word
        self._build_word_beginning_table(len(word_beginning))
        return self._word_beginning_tables[len(word_beginning)].get(word_beginning, [])

    def _build_word_beginning_table(self, length):
        if length not in self._word_beginning_tables:
            table = {}
            for word in get_progress_bar('building prefix lookup ')(self.index2word):
                if len(word) >= length:
                    ending = word[:length]
                    if ending not in table:
                        table[ending] = [word]
                    else:
                        table[ending].append(word)
            self._word_beginning_tables[length] = table

    @staticmethod
    def get(path):
        global WORD2VEC
        if WORD2VEC is None:
            WORD2VEC = FastWord2vec(path)
        return WORD2VEC
